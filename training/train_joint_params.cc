#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <thread>
#include <pthread.h>

#include <cmath>

#include "xalloc.h"
#include "llist.h"
#include "utils.h"
#include "train_utils.h"
#include "loader.h"
#include "infer.h"

#include "half.hpp"


#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.001f
#define PROGRESS_WIDTH 80

using half_float::half;


static bool verbose = false;

typedef struct {
  unsigned n_trees;       // Number of decision trees
  RDTree** forest;        // Decision trees

  uint32_t n_images;      // Number of training images
  int32_t  width;         // Width of training images
  int32_t  height;        // Height of training images
  uint8_t* label_images;  // Label images (row-major)
  half*    depth_images;  // Depth images (row-major)
  float**  inferred;      // Inferred label probabilities
  float*   weights;       // Pixel weighting for joint label groups

  uint8_t  n_joints;      // Number of joints
  char**   joint_names;   // Names of joints
  LList**  joint_map;     // Lists of which labels correspond to which joints
  float*   joints;        // List of joint positions for each image
  uint8_t  bg_label;      // Background label

  uint32_t n_bandwidths;  // Number of bandwidth values
  float*   bandwidths;    // Bandwidth values to test
  uint32_t n_thresholds;  // Number of probability thresholds
  float*   thresholds;    // Probability thresholds to test
  uint32_t n_offsets;     // Number of Z offsets
  float*   offsets;       // Z offsets to test

  uint32_t n_threads;     // Number of threads to use for work

  bool check_accuracy;    // Test the accuracy of joint inferrence
} TrainContext;

typedef struct {
  TrainContext*      ctx;              // Training context
  uint32_t           thread;           // Thread number
  float*             best_dist;        // Best mean distance from each joint
  float*             best_bandwidth;   // Best bandwidth per joint
  float*             best_threshold;   // Best threshold per joint
  float*             best_offset;      // Best offset per joint
  float*             accuracy;         // Accuracy of inference
  pthread_barrier_t* barrier ;         // Barrier to synchronise dependent work
} TrainThreadData;

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: train_joint_params <data dir> \\\n"
"                          <joint map> \\\n"
"                          <out_file> \\\n"
"                          [OPTIONS] \\\n"
"                          -- <tree file 1> [tree file 2] ...\n"
"Given a trained decision tree, train parameters for joint position proposal.\n"
"\n"
"  -l, --limit=NUMBER[,NUMBER] Limit training data to this many images.\n"
"                              Optionally, skip the first N images.\n"
"  -s, --shuffle               Shuffle order of training images\n"
"  -b, --bandwidths=MIN,MAX,N  Range of bandwidths to test\n"
"  -t, --thresholds=MIN,MAX,N  Range of probability thresholds to test\n"
"  -z, --offsets=MIN,MAX,N     Range of Z offsets to test\n"
"  -g, --background=NUMBER     Background label id (default: 0)\n"
"  -m, --threads=NUMBER        Number of threads to use (default: autodetect)\n"
"  -a, --accuracy              Report accuracy of joint inference\n"
"  -v, --verbose               Verbose output\n"
"  -h, --help                  Display this message\n");
}

/*static void
reproject(int32_t x, int32_t y, float depth, int32_t width, int32_t height,
          float vfov, float* out_point)
{
  if (!std::isnormal(depth))
    {
      return;
    }

  float half_width = width / 2.f;
  float half_height = height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = vfov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;
  //float hfov = atanf(tan_half_hfov) * 2;

  float s = (x / half_width) - 1.f;
  float t = (y / half_height) - 1.f;

  out_point[0] = (tan_half_hfov * depth) * s;
  out_point[1] = (tan_half_vfov * depth) * t;
  out_point[2] = depth;
}*/

static void*
thread_body(void* userdata)
{
  TrainThreadData* data = (TrainThreadData*)userdata;
  TrainContext* ctx = data->ctx;

  uint8_t n_labels = ctx->forest[0]->header.n_labels;

  // Generate probability tables and pixel weights, and possibly calculate
  // inference accuracy
  uint32_t images_per_thread =
    std::max((uint32_t)1, ctx->n_images / ctx->n_threads);
  uint32_t i_start = images_per_thread * data->thread;
  uint32_t i_end = std::min(ctx->n_images,
                            (data->thread == ctx->n_threads - 1) ?
                              ctx->n_images : i_start + images_per_thread);
  for (uint32_t i = i_start, idx = ctx->width * ctx->height * i_start;
       i < i_end; i++, idx += ctx->width * ctx->height)
    {
      ctx->inferred[i] = infer(ctx->forest, ctx->n_trees,
                               &ctx->depth_images[idx],
                               ctx->width, ctx->height);

      // Calculate pixel weight
      uint32_t pixel_idx = 0;
      uint32_t weight_idx = i * ctx->width * ctx->height * ctx->n_joints;
      for (int32_t y = 0; y < ctx->height; y++)
        {
          for (int32_t x = 0; x < ctx->width; x++, pixel_idx++)
            {
              float depth_2 =
                powf((float)ctx->depth_images[idx + pixel_idx], 2);

              for (uint8_t j = 0; j < ctx->n_joints; j++, weight_idx++)
                {
                  float pr = 0.f;
                  for (LList* node = ctx->joint_map[j]; node; node = node->next)
                    {
                      uint8_t label = (uint8_t)((uintptr_t)node->data);
                      pr += ctx->inferred[i][(pixel_idx * n_labels) + label];
                    }
                  ctx->weights[weight_idx] = pr * depth_2;
                }
            }
        }

      // Calculate inference accuracy if label images were specified
      if (ctx->check_accuracy)
        {
          uint32_t label_incidence[n_labels];
          uint32_t correct_label_incidence[n_labels];
          uint32_t label_idx = idx;

          /* NB: clang doesn't allow using an = {0} initializer with dynamic
           * sized arrays...
           */
          memset(label_incidence, 0, n_labels * sizeof(label_incidence[0]));
          memset(correct_label_incidence, 0,
                 n_labels * sizeof(correct_label_incidence[0]));

          for (int32_t y = 0; y < ctx->height; y++)
            {
              for (int32_t x = 0; x < ctx->width; x++, label_idx++)
                {
                  uint8_t actual_label = ctx->label_images[label_idx];

                  float best_pr = 0.f;
                  uint8_t inferred_label = 0;
                  for (uint8_t l = 0; l < n_labels; l++)
                    {
                      float pr =
                        ctx->inferred[i][((label_idx - idx) * n_labels) + l];
                      if (pr > best_pr)
                        {
                          best_pr = pr;
                          inferred_label = l;
                        }
                    }

                  label_incidence[actual_label] ++;
                  if (inferred_label == actual_label)
                    {
                      correct_label_incidence[inferred_label]++;
                    }
                }
            }

          uint8_t present_labels = 0;
          float accuracy = 0.f;
          for (uint8_t l = 0; l < n_labels; l++)
            {
              if (label_incidence[l] > 0)
                {
                  accuracy += correct_label_incidence[l] /
                              (float)label_incidence[l];
                  present_labels ++;
                }
            }
          accuracy /= (float)present_labels;

          *data->accuracy += accuracy;
        }
    }

  if (ctx->check_accuracy)
    {
      *data->accuracy /= (float)(i_end - i_start);
    }

  // Wait for all threads to finish
  pthread_barrier_wait(data->barrier);

  // Wait for main thread to output progress bar
  pthread_barrier_wait(data->barrier);

  // Use mean-shift to find the inferred joint positions, set them back into
  // the body using a range of thresholds/bandwidths/offsets, and record the
  // results
  uint32_t n_combos = ctx->n_bandwidths * ctx->n_thresholds * ctx->n_offsets;
  uint32_t combos_per_thread = std::max((uint32_t)1, n_combos / ctx->n_threads);
  uint32_t c_start = combos_per_thread * data->thread;
  uint32_t c_end = std::min(n_combos,
                            (data->thread == ctx->n_threads - 1) ?
                              n_combos : c_start + combos_per_thread);

  uint32_t bandwidth_stride = ctx->n_thresholds * ctx->n_offsets;

  uint32_t* n_pixels = (uint32_t*)xmalloc(ctx->n_joints * sizeof(uint32_t));
  float* points = (float*)
    xmalloc(ctx->n_joints * ctx->width * ctx->height * 3 * sizeof(float));
  float* weights = (float*)
    xmalloc(ctx->n_joints * ctx->width * ctx->height * sizeof(float));

  // Variables for reprojection of 2d point + depth
  float half_width = ctx->width / 2.f;
  float half_height = ctx->height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = ctx->forest[0]->header.fov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;
  //float hfov = atanf(tan_half_hfov) * 2;

  float root_2pi = sqrtf(2 * M_PI);

  float output_acc = 0;
  float output_freq = (c_end - c_start) /
    (PROGRESS_WIDTH / (float)ctx->n_threads);

  for (uint32_t c = c_start; c < c_end; c++)
    {
      uint32_t bandwidth_idx = c / bandwidth_stride;
      uint32_t threshold_idx = (c / ctx->n_offsets) % ctx->n_thresholds;
      uint32_t offset_idx = c % ctx->n_offsets;

      float bandwidth = ctx->bandwidths[bandwidth_idx];
      float threshold = ctx->thresholds[threshold_idx];
      float offset = ctx->offsets[offset_idx];

      /* NB: clang doesn't allow using an = {0} initializer with dynamic
       * sized arrays...
       */
      float acc_distance[ctx->n_joints];
      memset(acc_distance, 0, ctx->n_joints * sizeof(acc_distance[0]));

      for (uint32_t i = 0, idx = 0; i < ctx->n_images; i++)
        {
          // Gather pixels above the given threshold
          memset(n_pixels, 0, ctx->n_joints * sizeof(uint32_t));
          for (int32_t y = 0, sidx = 0; y < ctx->height; y++)
            {
              float t = -((y / half_height) - 1.f);
              for (int32_t x = 0; x < ctx->width; x++, idx++, sidx++)
                {
                  float s = (x / half_width) - 1.f;
                  float depth = (float)ctx->depth_images[idx];
                  if (!std::isnormal(depth))
                    {
                      continue;
                    }

                  for (uint8_t j = 0; j < ctx->n_joints; j++)
                    {
                      uint32_t joint_idx = j * ctx->width * ctx->height;
                      for (LList* node = ctx->joint_map[j]; node;
                           node = node->next)
                        {
                          uint8_t label = (uint8_t)((uintptr_t)node->data);
                          float label_pr =
                            ctx->inferred[i][(sidx * n_labels) + label];
                          if (label_pr >= threshold)
                            {
                              // Reproject point
                              points[(joint_idx + n_pixels[j]) * 3] =
                                (tan_half_hfov * depth) * s;
                              points[(joint_idx + n_pixels[j]) * 3 + 1] =
                                (tan_half_vfov * depth) * t;
                              points[(joint_idx + n_pixels[j]) * 3 + 2] =
                                depth;

                              // Store pixel weight (density)
                              weights[joint_idx + n_pixels[j]] =
                                ctx->weights[(idx * ctx->n_joints) + j];

                              n_pixels[j]++;
                              break;
                            }
                        }
                    }
                }
            }

          // Means shift to find joint modes
          for (uint8_t j = 0; j < ctx->n_joints; j++)
            {
              if (n_pixels[j] == 0)
                {
                  continue;
                }

              uint32_t joint_idx = j * ctx->width * ctx->height;
              for (uint32_t s = 0; s < N_SHIFTS; s++)
                {
                  float new_points[n_pixels[j] * 3];
                  bool moved = false;
                  for (uint32_t p = 0; p < n_pixels[j]; p++)
                    {
                      float* x = &points[(joint_idx + p) * 3];
                      float* nx = &new_points[p * 3];
                      float numerator[3] = { 0.f, };
                      float denominator = 0.f;
                      for (uint32_t n = 0; n < n_pixels[j]; n++)
                        {
                          float* xi = &points[(joint_idx + n) * 3];
                          float distance = sqrtf(pow(x[0] - xi[0], 2.f) +
                                                 pow(x[1] - xi[1], 2.f) +
                                                 pow(x[2] - xi[2], 2.f));

                          // Weighted gaussian kernel
                          float weight = weights[joint_idx + n] *
                            (1.f / (bandwidth * root_2pi)) *
                            expf(-0.5f * pow(distance / bandwidth, 2.f));

                          numerator[0] += weight * xi[0];
                          numerator[1] += weight * xi[1];
                          numerator[2] += weight * xi[2];

                          denominator += weight;
                        }

                      nx[0] = numerator[0] / denominator;
                      nx[1] = numerator[1] / denominator;
                      nx[2] = numerator[2] / denominator;

                      if (!moved &&
                          (fabs(nx[0] - x[0]) >= SHIFT_THRESHOLD ||
                           fabs(nx[1] - x[1]) >= SHIFT_THRESHOLD ||
                           fabs(nx[2] - x[2]) >= SHIFT_THRESHOLD))
                        {
                          moved = true;
                        }
                    }

                  memcpy((void*)&points[joint_idx * 3], (void*)new_points,
                         n_pixels[j] * 3 * sizeof(float));

                  if (!moved || s == N_SHIFTS - 1)
                    {
                      // Find the mode we're most confident of
                      float* last_point = &points[joint_idx * 3];
                      float confidence = 0;

                      float* best_point = last_point;
                      float best_confidence = 0;

                      //uint32_t unique_points = 1;

                      for (uint32_t p = 0; p < n_pixels[j]; p++)
                        {
                          float* point = &points[(joint_idx + p) * 3];
                          if (fabs(point[0]-last_point[0]) >= SHIFT_THRESHOLD ||
                              fabs(point[1]-last_point[1]) >= SHIFT_THRESHOLD ||
                              fabs(point[2]-last_point[2]) >= SHIFT_THRESHOLD)
                            {
                              if (confidence > best_confidence)
                                {
                                  best_point = last_point;
                                  best_confidence = confidence;
                                }
                              //unique_points++;
                              last_point = point;
                              confidence = 0;
                            }
                          confidence += weights[joint_idx + p];
                        }

                      // Offset into the body
                      best_point[2] += offset;

                      // Calculate distance from expected joint position
                      float* actual_joint =
                        &ctx->joints[((i * ctx->n_joints) + j) * 3];
                      // XXX: Current joint z positions are negated (and off?)
                      float distance =
                        sqrtf(pow(best_point[0] - actual_joint[0], 2.f) +
                              pow(best_point[1] - actual_joint[1], 2.f) +
                              pow(best_point[2] + actual_joint[2], 2.f));

                      // Accumulate
                      acc_distance[j] += distance;
                      break;
                    }
                }
            }
        }

      // See if this combination is better than the current best for any
      // particular joint
      for (uint8_t j = 0; j < ctx->n_joints; j++)
        {
          if (acc_distance[j] < data->best_dist[j])
            {
              data->best_dist[j] = acc_distance[j];
              data->best_bandwidth[j] = bandwidth;
              data->best_threshold[j] = threshold;
              data->best_offset[j] = offset;
            }
        }

      if (++output_acc >= output_freq)
        {
          fputc('x', stdout);
          fflush(stdout);
          output_acc -= output_freq;
        }
    }

  xfree(n_pixels);
  xfree(points);
  xfree(weights);

  pthread_exit(NULL);
}

static bool
read_three(char* string, float* value1, float* value2, uint32_t* value3)
{
  char* old_string = string;
  *value1 = strtof(old_string, &string);
  if (string == old_string)
    {
      return false;
    }

  old_string = string + 1;
  *value2 = strtof(old_string, &string);
  if (string == old_string)
    {
      return false;
    }

  old_string = string + 1;
  *value3 = (uint32_t)strtol(old_string, &string, 10);
  if (string == old_string || string[0] != '\0')
    {
      return false;
    }

  return true;
}

void
gen_range(float** data, float min, float max, uint32_t n)
{
  *data = (float*)xmalloc(n * sizeof(float));
  if (n == 1)
    {
      (*data)[0] = (max + min) / 2.f;
      return;
    }

  for (uint32_t i = 0; i < n; i++)
    {
      (*data)[i] = min + ((max - min) * i) / (float)(n - 1);
    }
}

int
main (int argc, char** argv)
{
  if (argc < 7)
    {
      print_usage(stderr);
      return 1;
    }

  // Variables for timing output
  TimeForDisplay since_begin, since_last;
  struct timespec begin, last, now;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  last = begin;

  // Set default parameters
  TrainContext ctx = { 0, };
  uint32_t limit = UINT32_MAX;
  uint32_t skip = 0;
  bool shuffle = false;
  ctx.n_bandwidths = 10;
  float min_bandwidth = 0.02f;
  float max_bandwidth = 0.08f;
  ctx.n_thresholds = 10;
  float min_threshold = 0.1f;
  float max_threshold = 0.5f;
  ctx.n_offsets = 10;
  float min_offset = 0.01f;
  float max_offset = 0.04f;
  ctx.n_threads = std::thread::hardware_concurrency();

  // Pass arguments
  char* data_dir = argv[1];
  char* joint_map_path = argv[2];
  char* out_filename = argv[3];

  char** tree_paths = NULL;
  for (int i = 5; i < argc; i++)
    {
      // All arguments should start with '-'
      if (argv[i][0] != '-')
        {
          print_usage(stderr);
          return 1;
        }
      char* arg = &argv[i][1];

      char param = '\0';
      char* value = NULL;
      if (arg[0] == '-')
        {
          // Argument was '--', signifying the end of parameters
          if (arg[1] == '\0')
            {
              if (i + 1 < argc)
                {
                  tree_paths = &argv[i + 1];
                  ctx.n_trees = argc - (i + 1);
                }
              break;
            }

          // Store the location of the value (if applicable)
          value = strchr(arg, '=');
          if (value)
            {
              value += 1;
            }

          // Check argument
          arg++;
          if (strstr(arg, "limit="))
            {
              param = 'l';
            }
          else if (strcmp(arg, "shuffle") == 0)
            {
              param = 's';
            }
          else if (strstr(arg, "bandwidths="))
            {
              param = 'b';
            }
          else if (strstr(arg, "thresholds="))
            {
              param = 't';
            }
          else if (strstr(arg, "offsets="))
            {
              param = 'z';
            }
          else if (strstr(arg, "accuracy"))
            {
              param = 'a';
            }
          else if (strstr(arg, "background="))
            {
              param = 'g';
            }
          else if (strstr(arg, "threads="))
            {
              param = 'm';
            }
          else if (strcmp(arg, "verbose") == 0)
            {
              param = 'v';
            }
          else if (strcmp(arg, "help") == 0)
            {
              param = 'h';
            }
          arg--;
        }
      else
        {
          if (arg[1] == '\0')
            {
              param = arg[0];
            }

          if (i + 1 < argc)
            {
              value = argv[i + 1];
            }
        }

      // Check for parameter-less options
      switch(param)
        {
        case 's':
          shuffle = true;
          continue;
        case 'v':
          verbose = true;
          continue;
        case 'h':
          print_usage(stdout);
          return 0;
        }

      // Now check for options that require parameters
      if (!value)
        {
          print_usage(stderr);
          return 1;
        }
      if (arg[0] != '-')
        {
          i++;
        }

      switch(param)
        {
        case 'l':
          limit = (uint32_t)strtol(value, &value, 10);
          if (value[0] != '\0')
            {
              skip = (uint32_t)strtol(value + 1, NULL, 10);
            }
          break;
        case 'b':
          read_three(value, &min_bandwidth, &max_bandwidth, &ctx.n_bandwidths);
          break;
        case 't':
          read_three(value, &min_threshold, &max_threshold, &ctx.n_thresholds);
          break;
        case 'z':
          read_three(value, &min_offset, &max_offset, &ctx.n_offsets);
          break;
        case 'a':
          ctx.check_accuracy = true;
          break;
        case 'g':
          ctx.bg_label = (uint8_t)atoi(value);
          break;
        case 'm':
          ctx.n_threads = (uint32_t)atoi(value);
          break;

        default:
          print_usage(stderr);
          return 1;
        }
    }

  if (!tree_paths)
    {
      print_usage(stderr);
      return 1;
    }

  printf("Loading decision forest...\n");
  ctx.forest = read_forest(tree_paths, ctx.n_trees);

  printf("Scanning training directories...\n");
  gather_train_data(data_dir, limit, skip, shuffle,
                    &ctx.n_images,
                    &ctx.n_joints,
                    &ctx.width, &ctx.height,
                    &ctx.depth_images,
                    ctx.check_accuracy ? &ctx.label_images : NULL,
                    &ctx.joints);

  // Note, there's a background label, so there ought to always be fewer joints
  // than labels. Maybe there are some situations where this might be desired
  // though, so just warn about it.
  if (ctx.n_joints >= ctx.forest[0]->header.n_labels)
    {
      fprintf(stderr, "WARNING: Joints exceeds labels (%u >= %u)\n",
              (uint32_t)ctx.n_joints, (uint32_t)ctx.forest[0]->header.n_labels);
    }

  printf("Loading joint map...\n");
  ctx.joint_map = (LList**)xcalloc(ctx.n_joints, sizeof(LList*));
  FILE* joint_map_file = fopen(joint_map_path, "r");
  if (!joint_map_file)
    {
      fprintf(stderr, "Error opening joint map\n");
      return 1;
    }

  ctx.joint_names = (char**)xmalloc(ctx.n_joints * sizeof(char*));
  for (uint8_t i = 0; i < ctx.n_joints; i++)
    {
      char buffer[256];
      if (!fgets(buffer, 256, joint_map_file))
        {
          fprintf(stderr, "Error reading joint %u\n", (uint32_t)i);
          return 1;
        }

      char* label_string = strchr(buffer, ',');
      if (!label_string || label_string == buffer)
        {
          fprintf(stderr, "Error reading joint %u name\n", (uint32_t)i);
          return 1;
        }
      ctx.joint_names[i] = strndup(buffer, label_string - buffer);

      while (label_string[0] != '\0' && label_string[0] != '\n')
        {
          label_string += 1;

          char* new_string;
          uint8_t label = strtol(label_string, &new_string, 10);
          if (new_string == label_string)
            {
              fprintf(stderr, "Error interpreting joint %u\n", (uint32_t)i);
              return 1;
            }

          if (label == ctx.bg_label)
            {
              fprintf(stderr, "Background label found in joint %u\n",
                      (uint32_t)i);
              return 1;
            }
          if (label >= ctx.forest[0]->header.n_labels)
            {
              fprintf(stderr, "Label out of range in joint %u\n", (uint32_t)i);
              return 1;
            }

          ctx.joint_map[i] = llist_insert_before(ctx.joint_map[i],
                               llist_new((void*)((uintptr_t)label)));
          label_string = new_string;
        }

      if (!ctx.joint_map[i])
        {
          fprintf(stderr, "No labels found for joint %u\n", (uint32_t)i);
          return 1;
        }
    }

  if (fclose(joint_map_file) != 0)
    {
      fprintf(stderr, "Error closing joint map file\n");
      return 1;
    }

  printf("Generating test parameters...\n");
  printf("%u bandwidths from %.3f to %.3f\n",
         ctx.n_bandwidths, min_bandwidth, max_bandwidth);
  printf("%u thresholds from %.3f to %.3f\n",
         ctx.n_thresholds, min_threshold, max_threshold);
  printf("%u offsets from %.3f to %.3f\n",
         ctx.n_offsets, min_offset, max_offset);
  gen_range(&ctx.bandwidths, min_bandwidth, max_bandwidth, ctx.n_bandwidths);
  gen_range(&ctx.thresholds, min_threshold, max_threshold, ctx.n_thresholds);
  gen_range(&ctx.offsets, min_offset, max_offset, ctx.n_offsets);


  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Beginning with %u threads...\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds,
         ctx.n_threads);

  pthread_barrier_t barrier;
  if (pthread_barrier_init(&barrier, NULL, ctx.n_threads + 1) != 0)
    {
      fprintf(stderr, "Error initialising thread barrier\n");
      return 1;
    }

  ctx.inferred = (float**)xmalloc(ctx.n_images * sizeof(float*));
  ctx.weights = (float*)xmalloc(ctx.n_images * ctx.width * ctx.height *
                                ctx.n_joints * sizeof(float));
  float* best_dists = (float*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
  std::fill(best_dists, best_dists + (ctx.n_joints * ctx.n_threads), FLT_MAX);
  float* best_bandwidths = (float*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
  float* best_thresholds = (float*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
  float* best_offsets = (float*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
  float* accuracies = (float*)xcalloc(ctx.n_threads, sizeof(float));
  pthread_t threads[ctx.n_threads];

  for (uint32_t i = 0; i < ctx.n_threads; i++)
    {
      TrainThreadData* thread_data = (TrainThreadData*)
        xcalloc(1, sizeof(TrainThreadData));
      thread_data->ctx = &ctx;
      thread_data->thread = i;
      thread_data->best_dist = &best_dists[i * ctx.n_joints];
      thread_data->best_bandwidth = &best_bandwidths[i * ctx.n_joints];
      thread_data->best_threshold = &best_thresholds[i * ctx.n_joints];
      thread_data->best_offset = &best_offsets[i * ctx.n_joints];
      thread_data->accuracy = &accuracies[i];
      thread_data->barrier = &barrier;

      if (pthread_create(&threads[i], NULL, thread_body,
                         (void*)thread_data) != 0)
        {
          fprintf(stderr, "Error creating thread\n");
          return 1;
        }
    }

  pthread_barrier_wait(&barrier);

  if (ctx.check_accuracy)
    {
      // We no longer need the label images
      xfree(ctx.label_images);

      // Calculate accuracy
      float accuracy = 0.f;
      uint32_t n_accuracies = std::min(ctx.n_threads, ctx.n_images);
      for (uint32_t i = 0; i < n_accuracies; i++)
        {
          accuracy += accuracies[i];
        }
      accuracy /= (float)n_accuracies;

      printf("Inference accuracy: %f\n", accuracy);
    }
  xfree(accuracies);

  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Waiting for mean shift...\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds);

  for (uint32_t i = 0; i < PROGRESS_WIDTH; i++)
    {
      printf("-");
    }
  printf("\n");

  // Let threads continue
  pthread_barrier_wait(&barrier);

  // Destroy barrier
  pthread_barrier_destroy(&barrier);

  // Wait for threads to finish
  for (uint32_t i = 0; i < ctx.n_threads; i++)
    {
      if (pthread_join(threads[i], NULL) != 0)
        {
          fprintf(stderr, "Error joining thread, trying to continue...\n");
        }
    }
  printf("\n");

  // Free memory we no longer need
  xfree(ctx.depth_images);
  xfree(ctx.weights);
  for (uint32_t i = 0; i < ctx.n_images; i++)
    {
      xfree(ctx.inferred[i]);
    }
  xfree(ctx.inferred);

  // Open output file
  FILE* output;
  JIPHeader header = { { 'J', 'I', 'P' }, JIP_VERSION, ctx.n_joints };
  if (!(output = fopen(out_filename, "wb")))
    {
      fprintf(stderr, "Failed to open output file\n");
    }
  else
    {
      if (fwrite(&header, sizeof(JIPHeader), 1, output) != 1)
        {
          fprintf(stderr, "Error writing header\n");

          fclose(output);
          output = NULL;
        }
    }

  // Find the best parameter combination and write to output file
  for (uint32_t j = 0; j < ctx.n_joints; j++)
    {
      for (uint32_t i = 1; i < ctx.n_threads; i++)
        {
          uint32_t idx = ctx.n_joints * i + j;
          if (best_dists[idx] < best_dists[j])
            {
              best_dists[j] = best_dists[idx];
              best_bandwidths[j] = best_bandwidths[idx];
              best_thresholds[j] = best_thresholds[idx];
              best_offsets[j] = best_offsets[idx];
            }
        }

      if (verbose || !output)
        {
          printf("Joint %d (%s): Mean distance: %.3fm\n"
                 "  Bandwidth: %f\n"
                 "  Threshold: %f\n"
                 "  Offset: %f\n",
                 j, ctx.joint_names[j], best_dists[j] / ctx.n_images,
                 best_bandwidths[j], best_thresholds[j], best_offsets[j]);
        }

      if (output)
        {
          if (fwrite(&best_bandwidths[j], sizeof(float), 1, output) != 1 ||
              fwrite(&best_thresholds[j], sizeof(float), 1, output) != 1 ||
              fwrite(&best_offsets[j], sizeof(float), 1, output) != 1)
            {
              fprintf(stderr, "Error writing output\n");
            }
        }
    }

  if (fclose(output) != 0)
    {
      fprintf(stderr, "Error closing output file\n");
    }

  // Free the last of the allocated memory
  xfree(best_dists);
  xfree(ctx.bandwidths);
  xfree(best_bandwidths);
  xfree(ctx.thresholds);
  xfree(best_thresholds);
  xfree(ctx.offsets);
  xfree(best_offsets);
  for (uint32_t i = 0; i < ctx.n_joints; i++)
    {
      xfree(ctx.joint_names[i]);
      llist_free(ctx.joint_map[i], NULL, NULL);
    }
  xfree(ctx.joint_names);
  xfree(ctx.joint_map);

  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Done!\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds);

  return 0;
}

