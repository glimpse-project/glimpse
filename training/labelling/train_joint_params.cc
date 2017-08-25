#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <thread>
#include <pthread.h>

#include "xalloc.h"
#include "llist.h"
#include "utils.h"
#include "train_utils.h"
#include "loader.h"
#include "infer.h"

static bool verbose = false;

typedef struct {
  uint8_t  n_trees;       // Number of decision trees
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
} TrainContext;

typedef struct {
  TrainContext*      ctx;              // Training context
  uint32_t           thread;           // Thread number
  float*             best_dist;        // Best mean distance from each joint
  uint32_t*          best_bandwidth;   // Best bandwidth index per joint
  uint32_t*          best_threshold;   // Best threshold index per joint
  uint32_t*          best_offset;      // Best offset index per joint
  float*             accuracy;         // Accuracy of inference
  pthread_barrier_t* barrier ;         // Barrier to synchronise dependent work
} TrainThreadData;

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: train_joint_params <depth dir> \\\n"
"                          <joint dir> \\\n"
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
"  -c, --labeldir=DIR          Label image directory\n"
"  -g, --background=NUMBER     Background label id (default: 0)\n"
"  -m, --threads=NUMBER        Number of threads to use (default: autodetect)\n"
"  -v, --verbose               Verbose output\n"
"  -h, --help                  Display this message\n");
}

static void*
thread_body(void* userdata)
{
  TrainThreadData* data = (TrainThreadData*)userdata;
  TrainContext* ctx = data->ctx;

  uint8_t n_labels = ctx->forest[0]->header.n_labels;

  // Generate probability tables and pixel weights, and possibly calculate
  // inference accuracy
  uint32_t images_per_thread = (ctx->n_images + ctx->n_threads - 1) /
                               ctx->n_threads;
  uint32_t i_start = images_per_thread * data->thread;
  uint32_t i_end = std::min(i_start + images_per_thread, ctx->n_images);
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
      if (ctx->label_images)
        {
          uint32_t label_incidence[n_labels] = { 0, };
          uint32_t correct_label_incidence[n_labels] = { 0, };
          uint32_t label_idx = idx;

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

  if (ctx->label_images)
    {
      *data->accuracy /= (float)(i_end - i_start);
    }

  pthread_barrier_wait(data->barrier);

  // 

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

  // Set default parameters
  TrainContext ctx = { 0, };
  uint32_t limit = UINT32_MAX;
  uint32_t skip = 0;
  bool shuffle = false;
  ctx.n_bandwidths = 50;
  float min_bandwidth = 0.015f;
  float max_bandwidth = 0.090f;
  ctx.n_thresholds = 50;
  float min_threshold = 0.1f;
  float max_threshold = 0.9f;
  ctx.n_offsets = 50;
  float min_offset = 0.025f;
  float max_offset = 0.075f;
  ctx.n_threads = std::thread::hardware_concurrency();

  // Pass arguments
  char* label_dir = NULL;
  char* depth_dir = argv[1];
  char* joint_dir = argv[2];
  char* joint_map_path = argv[3];
  char* out_filename = argv[4];

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
          else if (strstr(arg, "labeldir="))
            {
              param = 'c';
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
        case 'c':
          label_dir = value;
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
  gather_train_data(label_dir, depth_dir, joint_dir, limit, skip, shuffle,
                    &ctx.n_images, &ctx.n_joints, &ctx.width, &ctx.height,
                    &ctx.depth_images, &ctx.label_images, &ctx.joints);

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
  gen_range(&ctx.bandwidths, min_bandwidth, max_bandwidth, ctx.n_bandwidths);
  gen_range(&ctx.thresholds, min_threshold, max_threshold, ctx.n_thresholds);
  gen_range(&ctx.offsets, min_offset, max_offset, ctx.n_offsets);

  printf("Beginning with %u threads...\n", ctx.n_threads);
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
  uint32_t* best_bandwidths = (uint32_t*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(uint32_t));
  uint32_t* best_thresholds = (uint32_t*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(uint32_t));
  uint32_t* best_offsets = (uint32_t*)
    xmalloc(ctx.n_joints * ctx.n_threads * sizeof(uint32_t));
  float* accuracies = (float*)xcalloc(ctx.n_threads, sizeof(float));
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

      pthread_t thread;
      if (pthread_create(&thread, NULL, thread_body, (void*)thread_data) != 0)
        {
          fprintf(stderr, "Error creating thread\n");
          return 1;
        }
    }

  pthread_barrier_wait(&barrier);

  if (ctx.label_images)
    {
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

  return 0;
}

