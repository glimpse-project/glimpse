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
#ifdef __APPLE__
#include "pthread_barrier/pthread_barrier.h"
#endif

#include <cmath>
#include <atomic>
#include <vector>

#include "xalloc.h"
#include "rdt_tree.h"
#include "infer_labels.h"
#include "joints_inferrer.h"
#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_data.h"

#define JIP_VERSION 0
#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.001f


static bool verbose = false;

typedef struct {
    struct gm_logger *log;

    float    fov;           // Camera field of view

    int      n_trees;       // Number of decision trees
    RDTree** forest;        // Decision trees

    int      n_images;      // Number of training images
    int      width;         // Width of training images
    int      height;        // Height of training images
    float*   depth_images;  // Depth images

    // Inferred joints for every combination of tested parameters (stored in
    // combination-major order)
    InferredJoints** inferred_joints;

    int      n_joints;      // Number of joints
    JSON_Value* joint_map;  // Map between joints and labels
    struct joints_inferrer* joints_inferrer;

    float*   joints;        // List of joint positions for each image

    int      n_bandwidths;  // Number of bandwidth values
    float*   bandwidths;    // Bandwidth values to test
    int      n_thresholds;  // Number of probability thresholds
    float*   thresholds;    // Probability thresholds to test
    int      n_offsets;     // Number of Z offsets
    float*   offsets;       // Z offsets to test

    bool     fast;          // Train for infer_joints_fast()
    int      n_threads;     // Number of threads to use for work
    std::atomic<int> progress; // Number of combinations processed
} TrainContext;

typedef struct {
    TrainContext*      ctx;              // Training context
    int                thread;           // Thread number
    float*             best_dist;        // Best mean distance from each joint
    float*             best_bandwidth;   // Best bandwidth per joint
    float*             best_threshold;   // Best threshold per joint
    float*             best_offset;      // Best offset per joint
    pthread_barrier_t* barrier ;         // Barrier to synchronise dependent work
} TrainThreadData;

typedef struct {
    int32_t hours;
    int32_t minutes;
    int32_t seconds;
} TimeForDisplay;

static TimeForDisplay
get_time_for_display(struct timespec* begin, struct timespec* end)
{
    uint32_t elapsed;
    TimeForDisplay display;

    elapsed = (end->tv_sec - begin->tv_sec);
    elapsed += (end->tv_nsec - begin->tv_nsec) / 1000000000;

    display.seconds = elapsed % 60;
    display.minutes = elapsed / 60;
    display.hours = display.minutes / 60;
    display.minutes = display.minutes % 60;

    return display;
}

static void
print_usage(FILE* stream)
{
    fprintf(stream,
"Usage: train_joint_params <data dir> \\\n"
"                          <index name> \\\n"
"                          <joint map> \\\n"
"                          <out_file.json> \\\n"
"                          [OPTIONS] \\\n"
"                          -- <tree file 1> [tree file 2] ...\n"
"Given a trained decision tree, train parameters for joint position proposal.\n"
"\n"
"  -b, --bandwidths=MIN,MAX,N  Range of bandwidths to test\n"
"  -t, --thresholds=MIN,MAX,N  Range of probability thresholds to test\n"
"  -z, --offsets=MIN,MAX,N     Range of Z offsets to test\n"
"  -f, --fast                  Train for infer_joints_fast (bandwidth is ignored)\n"
"  -j, --threads=NUMBER        Number of threads to use (default: autodetect)\n"
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

    int last_output = -1;

    int n_labels = ctx->forest[0]->header.n_labels;
    float bg_depth = ctx->forest[0]->header.bg_depth;

    int n_combos = ctx->n_bandwidths * ctx->n_thresholds * ctx->n_offsets;
    int combos_per_thread = std::max(1, n_combos / ctx->n_threads);
    int c_start = combos_per_thread * data->thread;
    int c_end = std::min(n_combos,
                         (data->thread == ctx->n_threads - 1) ?
                         n_combos : c_start + combos_per_thread);
    int bandwidth_stride = ctx->n_thresholds * ctx->n_offsets;

    // For a subset of all depth images, generate the weights and probability
    // table and then test each combination of parameters and store the
    // resulting joints.
    int images_per_thread =
        std::max(1, ctx->n_images / ctx->n_threads);
    int i_start = images_per_thread * data->thread;
    int i_end = std::min(ctx->n_images,
                         (data->thread == ctx->n_threads - 1) ?
                         ctx->n_images : i_start + images_per_thread);

    std::vector<float> pr_table(ctx->width * ctx->height * n_labels);
    std::vector<float> weights(ctx->width * ctx->height * n_labels);

    for (int i = i_start, idx = ctx->width * ctx->height * i_start;
         i < i_end; i++, idx += ctx->width * ctx->height)
    {
        if (data->thread == 0) {
            int output = (int)(ctx->progress / (float)ctx->n_images * 100.f);
            if (output != last_output) {
                last_output = output;
                printf("%03d%%\r", output);
                fflush(stdout);
            }
        }

        float *depth_image = &ctx->depth_images[idx];

        infer_labels(ctx->log,
                     ctx->forest,
                     ctx->n_trees,
                     depth_image,
                     ctx->width, ctx->height,
                     pr_table.data(),
                     false, // don't use multi-threaded inference
                     false); // don't combine horizontal flipped results

        joints_inferrer_calc_pixel_weights(ctx->joints_inferrer,
                                           &ctx->depth_images[idx],
                                           pr_table.data(),
                                           ctx->width, ctx->height,
                                           n_labels,
                                           weights.data());

        // For each combination this thread is processing, infer the joint
        // positions for this depth image.
        for (int c = 0; c < n_combos; c++) {
            int bandwidth_idx = c / bandwidth_stride;
            int threshold_idx = (c / ctx->n_offsets) % ctx->n_thresholds;
            int offset_idx = c % ctx->n_offsets;

            float bandwidth = ctx->bandwidths[bandwidth_idx];
            float threshold = ctx->thresholds[threshold_idx];
            float offset = ctx->offsets[offset_idx];

            JIParam params[ctx->n_joints];
            for (int j = 0; j < ctx->n_joints; ++j)
            {
                params[j].bandwidth = bandwidth;
                params[j].threshold = threshold;
                params[j].offset = offset;
            }

            if (ctx->fast) {
                ctx->inferred_joints[(i * n_combos) + c] =
                    joints_inferrer_infer_fast(ctx->joints_inferrer,
                                               depth_image,
                                               pr_table.data(),
                                               weights.data(),
                                               ctx->width, ctx->height,
                                               n_labels,
                                               ctx->forest[0]->header.fov,
                                               params);
            } else {
                ctx->inferred_joints[(i * n_combos) + c] =
                    joints_inferrer_infer(ctx->joints_inferrer,
                                          depth_image,
                                          pr_table.data(),
                                          weights.data(),
                                          ctx->width, ctx->height,
                                          bg_depth,
                                          n_labels,
                                          ctx->forest[0]->header.fov,
                                          params);
            }
        }

        ctx->progress++;
    }

    // Wait for all threads to finish
    pthread_barrier_wait(data->barrier);

    // Wait for main thread to write a newline
    pthread_barrier_wait(data->barrier);

    // Loop over each bandwidth/threshold/offset combination and test to see
    // which combination gives the best results for inference on each joint.

    last_output = -1;

    for (int c = c_start; c < c_end; c++)
    {
        if (data->thread == 0) {
            int output = (int)(ctx->progress / (float)n_combos * 100.f);
            if (output != last_output) {
                last_output = output;
                printf("%03d%%\r", output);
                fflush(stdout);
            }
        }

        int bandwidth_idx = c / bandwidth_stride;
        int threshold_idx = (c / ctx->n_offsets) % ctx->n_thresholds;
        int offset_idx = c % ctx->n_offsets;

        float bandwidth = ctx->bandwidths[bandwidth_idx];
        float threshold = ctx->thresholds[threshold_idx];
        float offset = ctx->offsets[offset_idx];

        /* NB: clang doesn't allow using an = {0} initializer with dynamic
         * sized arrays...
         */
        float acc_distance[ctx->n_joints];
        memset(acc_distance, 0, ctx->n_joints * sizeof(acc_distance[0]));

        for (int i = 0; i < ctx->n_images; i++)
        {
            InferredJoints* result = ctx->inferred_joints[(i * n_combos) + c];

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
                acc_distance[j] += std::min(10.f, distance);
            }

            // Free joint positions
            joints_inferrer_free_joints(ctx->joints_inferrer, result);
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

        ctx->progress++;
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

    for (int i = 0; i < n; i++) {
        (*data)[i] = min + ((max - min) * i) / (float)(n - 1);
    }
}

int
main(int argc, char** argv)
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

    ctx.log = gm_logger_new(NULL, NULL);

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
            else if (strcmp(arg, "fast") == 0)
            {
                param = 'f';
            }
            else if (strstr(arg, "threads="))
            {
                param = 'j';
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
        case 'f':
            ctx.fast = true;
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
        case 'b':
            read_three(value, &min_bandwidth, &max_bandwidth, &ctx.n_bandwidths);
            break;
        case 't':
            read_three(value, &min_threshold, &max_threshold, &ctx.n_thresholds);
            break;
        case 'z':
            read_three(value, &min_offset, &max_offset, &ctx.n_offsets);
            break;
        case 'j':
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
    RDTree *forest[ctx.n_trees];
    JSON_Value *forest_js[ctx.n_trees];
    for (int i = 0; i < ctx.n_trees; i++) {
        char *tree_path = tree_paths[i];

        printf("> Loading %s...\n", tree_path);
        forest_js[i] = json_parse_file(tree_path);
        gm_assert(ctx.log, forest_js[i] != NULL, "Failed to parse %s as JSON", tree_path);

        forest[i] = rdt_tree_load_from_json(ctx.log,
                                            forest_js[i],
                                            false, // don't load incomplete trees
                                            NULL); // abort on error
    }
    ctx.forest = forest;

    printf("Scanning training directories...\n");
    JSON_Value *meta =
        gm_data_load_simple(ctx.log,
                            data_dir,
                            index_name,
                            joint_map_path,
                            &ctx.n_images,
                            &ctx.n_joints,
                            &ctx.width, &ctx.height,
                            &ctx.depth_images,
                            NULL, // skip label images
                            &ctx.joints,
                            NULL); // simply abort on error

    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");
    ctx.fov = json_object_get_number(camera, "vertical_fov");
    json_value_free(meta);
    meta = NULL;
    camera = NULL;

    printf("Loading joint map...\n");
    ctx.joint_map = json_parse_file(joint_map_path);
    if (!ctx.joint_map)
    {
        fprintf(stderr, "Failed to load joint map %s\n", joint_map_path);
        return 1;
    }

    // Joints are derived from labels so it's unlikely to ever make sense to
    // have more joints defined than labels.
    gm_assert(ctx.log, ctx.n_joints < ctx.forest[0]->header.n_labels,
              "More joints defined than labels");

    ctx.joints_inferrer = joints_inferrer_new(ctx.log,
                                              ctx.joint_map,
                                              NULL); // abort on error

    printf("Generating test parameters...\n");
    if (ctx.fast) {
        ctx.n_bandwidths = 1;
    } else {
        printf("%u bandwidths from %.3f to %.3f\n",
               ctx.n_bandwidths, min_bandwidth, max_bandwidth);
    }
    gen_range(&ctx.bandwidths, min_bandwidth, max_bandwidth, ctx.n_bandwidths);

    printf("%u thresholds from %.3f to %.3f\n",
           ctx.n_thresholds, min_threshold, max_threshold);
    gen_range(&ctx.thresholds, min_threshold, max_threshold, ctx.n_thresholds);

    printf("%u offsets from %.3f to %.3f\n",
           ctx.n_offsets, min_offset, max_offset);
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

    size_t n_combos = (size_t)ctx.n_bandwidths * ctx.n_thresholds *
                      ctx.n_offsets;
    ctx.inferred_joints = (InferredJoints**)xmalloc(n_combos * ctx.n_images *
                                                    sizeof(InferredJoints*));
    float* best_dists = (float*)
        xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
    std::fill(best_dists, best_dists + (ctx.n_joints * ctx.n_threads), FLT_MAX);
    float* best_bandwidths = (float*)
        xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
    float* best_thresholds = (float*)
        xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
    float* best_offsets = (float*)
        xmalloc(ctx.n_joints * ctx.n_threads * sizeof(float));
    pthread_t threads[ctx.n_threads];

    printf("(%02d:%02d:%02d / %02d:%02d:%02d) Running joint inference...\n",
           since_begin.hours, since_begin.minutes, since_begin.seconds,
           since_last.hours, since_last.minutes, since_last.seconds);
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
        thread_data->barrier = &barrier;

        if (pthread_create(&threads[i], NULL, thread_body,
                           (void*)thread_data) != 0)
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    pthread_barrier_wait(&barrier);
    printf("100%%\n");
    ctx.progress = 0;

    clock_gettime(CLOCK_MONOTONIC, &now);
    since_begin = get_time_for_display(&begin, &now);
    since_last = get_time_for_display(&last, &now);
    last = now;
    printf("(%02d:%02d:%02d / %02d:%02d:%02d) Testing combinations...\n",
           since_begin.hours, since_begin.minutes, since_begin.seconds,
           since_last.hours, since_last.minutes, since_last.seconds);

    // Let threads continue
    pthread_barrier_wait(&barrier);

    // Destroy barrier
    pthread_barrier_destroy(&barrier);

    // Wait for threads to finish
    for (int i = 0; i < ctx.n_threads; i++)
    {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread, trying to continue...\n");
        }
    }
    printf("100%%\n");

    // Free memory we no longer need
    xfree(ctx.inferred_joints);

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

            gm_assert(ctx.log, best_dists[j] < FLT_MAX,
                      "Uninitialised best distance found");

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
        JIPHeader header = {
            { 'J', 'I', 'P' },
            JIP_VERSION,
            (uint8_t)ctx.n_joints
        };
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

    for (int i = 0; i < ctx.n_trees; i++) {
        rdt_tree_destroy(ctx.forest[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    since_begin = get_time_for_display(&begin, &now);
    since_last = get_time_for_display(&last, &now);
    last = now;
    printf("(%02d:%02d:%02d / %02d:%02d:%02d) Done!\n",
           since_begin.hours, since_begin.minutes, since_begin.seconds,
           since_last.hours, since_last.minutes, since_last.seconds);

    return 0;
}

