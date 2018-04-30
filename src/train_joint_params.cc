/*
 * Copyright (C) 2017 Glimp IP Ltd
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
#include "parson.h"

#include "half.hpp"


#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.001f
#define PROGRESS_WIDTH 80

using half_float::half;


static bool verbose = false;

typedef struct {
  float    fov;           // Camera field of view

  int      n_trees;       // Number of decision trees
  RDTree** forest;        // Decision trees

  int      n_images;      // Number of training images
  int      width;         // Width of training images
  int      height;        // Height of training images
  uint8_t* label_images;  // Label images (row-major)
  half*    depth_images;  // Depth images (row-major)
  float**  inferred;      // Inferred label probabilities
  float*   weights;       // Pixel weighting for joint label groups

  int      n_joints;      // Number of joints
  JSON_Value* joint_map;  // Map between joints and labels
  float*   joints;        // List of joint positions for each image

  int      n_bandwidths;  // Number of bandwidth values
  float*   bandwidths;    // Bandwidth values to test
  int      n_thresholds;  // Number of probability thresholds
  float*   thresholds;    // Probability thresholds to test
  int      n_offsets;     // Number of Z offsets
  float*   offsets;       // Z offsets to test

  int      n_threads;     // Number of threads to use for work

  bool check_accuracy;    // Test the accuracy of joint inferrence
} TrainContext;

typedef struct {
  TrainContext*      ctx;              // Training context
  int                thread;           // Thread number
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
"                          <index name> \\\n"
"                          <joint map> \\\n"
"                          <out_file> \\\n"
"                          [OPTIONS] \\\n"
"                          -- <tree file 1> [tree file 2] ...\n"
"Given a trained decision tree, train parameters for joint position proposal.\n"
"\n"
"  -b, --bandwidths=MIN,MAX,N  Range of bandwidths to test\n"
"  -t, --thresholds=MIN,MAX,N  Range of probability thresholds to test\n"
"  -z, --offsets=MIN,MAX,N     Range of Z offsets to test\n"
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

  int n_labels = ctx->forest[0]->header.n_labels;

  // Generate probability tables and pixel weights, and possibly calculate
  // inference accuracy
  int images_per_thread =
    std::max(1, ctx->n_images / ctx->n_threads);
  int i_start = images_per_thread * data->thread;
  int i_end = std::min(ctx->n_images,
                       (data->thread == ctx->n_threads - 1) ?
                       ctx->n_images : i_start + images_per_thread);
  for (int i = i_start, idx = ctx->width * ctx->height * i_start;
       i < i_end; i++, idx += ctx->width * ctx->height)
    {
      ctx->inferred[i] = infer_labels<half>(ctx->forest, ctx->n_trees,
                                            &ctx->depth_images[idx],
                                            ctx->width, ctx->height);

      // Calculate pixel weight
      int weight_idx = i * ctx->width * ctx->height * ctx->n_joints;
      calc_pixel_weights<half>(&ctx->depth_images[idx], ctx->inferred[i],
                               ctx->width, ctx->height, n_labels,
                               ctx->joint_map,
                               &ctx->weights[weight_idx]);

      // Calculate inference accuracy if label images were specified
      if (ctx->check_accuracy)
        {
          int label_incidence[n_labels];
          int correct_label_incidence[n_labels];
          int label_idx = idx;

          /* NB: clang doesn't allow using an = {0} initializer with dynamic
           * sized arrays...
           */
          memset(label_incidence, 0, n_labels * sizeof(label_incidence[0]));
          memset(correct_label_incidence, 0,
                 n_labels * sizeof(correct_label_incidence[0]));

          for (int y = 0; y < ctx->height; y++)
            {
              for (int x = 0; x < ctx->width; x++, label_idx++)
                {
                  int actual_label = (int)ctx->label_images[label_idx];

                  float best_pr = 0.f;
                  int inferred_label = 0;
                  for (int l = 0; l < n_labels; l++)
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

          int present_labels = 0;
          float accuracy = 0.f;
          for (int l = 0; l < n_labels; l++)
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

  // Loop over each bandwidth/threshold/offset combination and test to see
  // which combination gives the best results for inference on each joint.
  int n_combos = ctx->n_bandwidths * ctx->n_thresholds * ctx->n_offsets;
  int combos_per_thread = std::max(1, n_combos / ctx->n_threads);
  int c_start = combos_per_thread * data->thread;
  int c_end = std::min(n_combos,
                       (data->thread == ctx->n_threads - 1) ?
                       n_combos : c_start + combos_per_thread);

  int bandwidth_stride = ctx->n_thresholds * ctx->n_offsets;

  float output_acc = 0;
  float output_freq = (c_end - c_start) /
    (PROGRESS_WIDTH / (float)ctx->n_threads);

  for (int c = c_start; c < c_end; c++)
    {
      int bandwidth_idx = c / bandwidth_stride;
      int threshold_idx = (c / ctx->n_offsets) % ctx->n_thresholds;
      int offset_idx = c % ctx->n_offsets;

      float bandwidth = ctx->bandwidths[bandwidth_idx];
      float threshold = ctx->thresholds[threshold_idx];
      float offset = ctx->offsets[offset_idx];

      JIParam params[ctx->n_joints];
      for (int i = 0; i < ctx->n_joints; i++)
        {
          params[i].bandwidth = bandwidth;
          params[i].threshold = threshold;
          params[i].offset = offset;
        }

      /* NB: clang doesn't allow using an = {0} initializer with dynamic
       * sized arrays...
       */
      float acc_distance[ctx->n_joints];
      memset(acc_distance, 0, ctx->n_joints * sizeof(acc_distance[0]));

      for (int i = 0; i < ctx->n_images; i++)
        {
          int depth_idx = i * ctx->width * ctx->height;
          int weight_idx = depth_idx * ctx->n_joints;

          half* depth_image = &ctx->depth_images[depth_idx];
          float* pr_table = ctx->inferred[i];
          float* weights = &ctx->weights[weight_idx];

          // Get joint positions
          InferredJoints* result =
            infer_joints<half>(depth_image, pr_table, weights,
                               ctx->width, ctx->height, n_labels,
                               ctx->joint_map,
                               ctx->forest[0]->header.fov,
                               params);

          // Calculate distance from expected joint position and accumulate
          for (int j = 0; j < ctx->n_joints; j++)
            {
              if (!result->joints[j])
                {
                  // If there's no predicted joint, just add a large number to
                  // the accumulated distance. Note that distances are in
                  // meters, so 10 is pretty large.
                  acc_distance[j] += 10.f;
                  continue;
                }

              Joint* inferred_joint = (Joint*)result->joints[j]->data;
              float* actual_joint =
                &ctx->joints[((i * ctx->n_joints) + j) * 3];

              // XXX: Current joint z positions are negated
              float distance =
                sqrtf(powf(inferred_joint->x - actual_joint[0], 2.f) +
                      powf(inferred_joint->y - actual_joint[1], 2.f) +
                      powf(inferred_joint->z + actual_joint[2], 2.f));

              // Accumulate
              acc_distance[j] += distance;
            }

          // Free joint positions
          free_joints(result);
        }

      // See if this combination is better than the current best for any
      // particular joint
      for (int j = 0; j < ctx->n_joints; j++)
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

  xfree(data);
  pthread_exit(NULL);
}

static bool
read_three(char* string, float* value1, float* value2, int* value3)
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
  *value3 = (int)strtol(old_string, &string, 10);
  if (string == old_string || string[0] != '\0')
    {
      return false;
    }

  return true;
}

void
gen_range(float** data, float min, float max, int n)
{
  *data = (float*)xmalloc(n * sizeof(float));
  if (n == 1)
    {
      (*data)[0] = (max + min) / 2.f;
      return;
    }

  for (int i = 0; i < n; i++)
    {
      (*data)[i] = min + ((max - min) * i) / (float)(n - 1);
    }
}

int
main (int argc, char** argv)
{
  if (argc < 5)
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
  char* index_name = argv[2];
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
          if (strstr(arg, "bandwidths="))
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
        case 'm':
          ctx.n_threads = atoi(value);
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
  ctx.forest = read_forest((const char**)tree_paths, ctx.n_trees);

  printf("Scanning training directories...\n");
  gather_train_data(data_dir,
                    index_name,
                    joint_map_path,
                    &ctx.n_images,
                    &ctx.n_joints,
                    &ctx.width, &ctx.height,
                    &ctx.depth_images,
                    ctx.check_accuracy ? &ctx.label_images : NULL,
                    &ctx.joints,
                    NULL, // n labels
                    &ctx.fov);

  // Note, there's a background label, so there ought to always be fewer joints
  // than labels. Maybe there are some situations where this might be desired
  // though, so just warn about it.
  if (ctx.n_joints >= ctx.forest[0]->header.n_labels)
    {
      fprintf(stderr, "WARNING: Joints exceeds labels (%d >= %u)\n",
              ctx.n_joints, (int)ctx.forest[0]->header.n_labels);
    }

  printf("Loading joint map...\n");
  ctx.joint_map = json_parse_file(joint_map_path);
  if (!ctx.joint_map)
    {
      fprintf(stderr, "Failed to load joint map %s\n", joint_map_path);
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

  for (int i = 0; i < ctx.n_threads; i++)
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
      int n_accuracies = std::min(ctx.n_threads, ctx.n_images);
      for (int i = 0; i < n_accuracies; i++)
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

  for (int i = 0; i < PROGRESS_WIDTH; i++)
    {
      printf("-");
    }
  printf("\n");

  // Let threads continue
  pthread_barrier_wait(&barrier);

  // Destroy barrier
  pthread_barrier_destroy(&barrier);

  // Wait for threads to finish
  for (int i = 0; i < ctx.n_threads; i++)
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
  for (int i = 0; i < ctx.n_images; i++)
    {
      xfree(ctx.inferred[i]);
    }
  xfree(ctx.inferred);

  // Open output file

  const char *ext;
  if ((ext = strstr(out_filename, ".json")) && ext[5] == '\0')
    {
      JSON_Value* js_root = json_value_init_object();

      json_object_set_number(json_object(js_root), "n_joints", ctx.n_joints);
      JSON_Value* js_params = json_value_init_array();

      for (int j = 0; j < ctx.n_joints; j++)
        {
          for (int i = 1; i < ctx.n_threads; i++)
            {
              int idx = ctx.n_joints * i + j;
              if (best_dists[idx] < best_dists[j])
                {
                  best_dists[j] = best_dists[idx];
                  best_bandwidths[j] = best_bandwidths[idx];
                  best_thresholds[j] = best_thresholds[idx];
                  best_offsets[j] = best_offsets[idx];
                }
            }

          JSON_Object *mapping = json_array_get_object(json_array(ctx.joint_map), j);
          const char *joint_name = json_object_get_string(mapping, "joint");

          JSON_Value* js_param = json_value_init_object();
          json_object_set_string(json_object(js_param), "name", joint_name);
          json_object_set_number(json_object(js_param), "bandwidth", best_bandwidths[j]);
          json_object_set_number(json_object(js_param), "threshold", best_thresholds[j]);
          json_object_set_number(json_object(js_param), "offset", best_offsets[j]);

          json_array_append_value(json_array(js_params), js_param);
        }

      json_object_set_value(json_object(js_root), "params", js_params);

      json_serialize_to_file_pretty(js_root, out_filename);
      json_value_free(js_root);
    }
  else
    {
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
      for (int j = 0; j < ctx.n_joints; j++)
        {
          for (int i = 1; i < ctx.n_threads; i++)
            {
              int idx = ctx.n_joints * i + j;
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
              JSON_Object *mapping = json_array_get_object(json_array(ctx.joint_map), j);
              const char *joint_name = json_object_get_string(mapping, "joint");

              printf("Joint %d (%s): Mean distance: %.3fm\n"
                     "  Bandwidth: %f\n"
                     "  Threshold: %f\n"
                     "  Offset: %f\n",
                     j, joint_name, best_dists[j] / ctx.n_images,
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
    }

  // Free the last of the allocated memory
  xfree(best_dists);
  xfree(ctx.bandwidths);
  xfree(best_bandwidths);
  xfree(ctx.thresholds);
  xfree(best_thresholds);
  xfree(ctx.offsets);
  xfree(best_offsets);
  json_value_free(ctx.joint_map);
  xfree(ctx.joints);
  free_forest(ctx.forest, ctx.n_trees);

  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Done!\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds);

  return 0;
}

