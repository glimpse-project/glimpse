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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <inttypes.h>

#include <random>
#include <thread>
#include <queue>

#include "half.hpp"

#include "xalloc.h"
#include "utils.h"
#include "rdt_tree.h"
#include "train_utils.h"

#include "glimpse_rdt.h"
#include "glimpse_log.h"
#include "glimpse_properties.h"

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "rdt"

using half_float::half;

static const char *interrupt_reason;
static bool interrupted;

typedef struct {
    int       id;              // Unique id to place the node a tree.
    int       n_pixels;        // Number of pixels that have reached this node.
    Int3D*    pixels;          // A list of pixel pairs and image indices.
} NodeTrainData;

/* Work submitted for the thread pool to process... */
struct work {
    NodeTrainData *node;
    int            uv_start;
    int            uv_end;
};

#define MAX_LABELS 40
struct result {
    float best_gain;
    int best_uv;
    int best_threshold;
    int n_lr_pixels[2];
    float nhistogram[MAX_LABELS];
    uint64_t duration;
};

struct thread_metrics {
    uint64_t duration;

    float idle_percent;
    float accumulation_percent;
    float ranking_percent;

    float nodes_per_second;
    int images_per_second;
    int pixels_per_second;
    int uvs_per_second;
    int thresholds_per_second;
};

struct thread_state {
    struct gm_rdt_context_impl* ctx;
    int idx;

    pthread_t thread;

    uint64_t last_metrics_log;

    /* Note: These metrics are reset at the start of each tree level */
    struct {
        uint64_t start;

        uint64_t idle_duration;

        uint64_t accumulation_time;
        uint64_t n_pixels_accumulated;
        uint64_t n_images_accumulated;
        uint64_t n_nodes;

        uint64_t gain_ranking_time;
    } metrics;

    std::vector<thread_metrics> per_depth_metrics;
};

struct aggregate_metrics {
    uint64_t duration;

    float wait_percent;
    float scheduling_percent;

    float nodes_per_second;
    int images_per_second;
    int pixels_per_second;
    int uvs_per_second;
    int thresholds_per_second;

    int pixels_per_node;
};

struct gm_rdt_context_impl {
    struct gm_logger* log;

    JSON_Value* data_meta;
    JSON_Value* record;
    JSON_Value* history;

    char*    reload;        // Reload and continue training with pre-existing tree
    bool     verbose;       // Verbose logging
    bool     profile;       // Verbose profiling
    bool     pretty;        // Pretty JSON output
    int      seed;          // Seed for RNG

    char*    data_dir;      // Location of training data
    char*    index_name;    // Name of the frame index like <name>.index to read
    char*    out_filename;  // Filename of tree (.json or .rdt) to write

    int      width;         // Width of training images
    int      height;        // Height of training images
    float    fov;           // Camera field of view
    int      n_labels;      // Number of labels in label images

    int      n_images;      // Number of training images
    uint8_t* label_images;  // Label images (row-major)
    half*    depth_images;  // Depth images (row-major)

    int      n_uvs;         // Number of combinations of u,v pairs
    float    uv_range;      // Range of u,v combinations to generate
    int      n_thresholds;  // The number of thresholds
    float    threshold_range;       // Range of thresholds to test
    int      max_depth;     // Maximum depth to train to
    int      max_nodes;     // Maximum number of nodes to train - used for debug
                            // and testing to trigger an early exit.
    int      n_pixels;      // Number of pixels to sample
    UVPair*  uvs;           // A list of uv pairs to test
    float*   thresholds;    // A list of thresholds to test

    int      n_threads;     // How many threads to spawn for training

    int      n_nodes_trained;   // The number of nodes trained so far

    std::queue<NodeTrainData> train_queue;

    std::vector<thread_state> thread_pool;

    // Queue of work for thread pool
    pthread_mutex_t     work_queue_lock;
    pthread_cond_t      work_queue_changed;
    std::queue<work>    work_queue;

    // Results computed by the worker threads
    pthread_mutex_t     results_lock;
    pthread_cond_t      results_changed;
    std::vector<result> results;

    std::vector<Node>   tree; // The decision tree being built
    std::vector<float>  tree_histograms; // label histograms for leaf nodes

    std::vector<int>    root_pixel_histogram; // label histogram for initial pixels
    std::vector<float>  root_pixel_nhistogram; // normalized histogram for initial pixels

    /* Note: These metrics are reset at the start of each tree level */
    struct {
        uint64_t start;
        uint64_t wait_duration;
    } metrics;
    std::vector<aggregate_metrics> per_depth_metrics;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
};


static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static float
get_format_duration(uint64_t duration_ns)
{
    if (duration_ns > 1000000000ULL*60*60)
        return duration_ns / (1e9 * 60.0 * 60.0);
    else if (duration_ns > 1000000000ULL*60)
        return duration_ns / (1e9 * 60.0);
    else if (duration_ns > 1000000000)
        return duration_ns / 1e9;
    else if (duration_ns > 1000000)
        return duration_ns / 1e6;
    else if (duration_ns > 1000)
        return duration_ns / 1e3;
    else
        return duration_ns;
}

static char *
get_format_duration_suffix(uint64_t duration_ns)
{
    if (duration_ns > 1000000000ULL*60*60)
        return (char *)"hr";
    else if (duration_ns > 1000000000ULL*60)
        return (char *)"min";
    else if (duration_ns > 1000000000)
        return (char *)"s";
    else if (duration_ns > 1000000)
        return (char *)"ms";
    else if (duration_ns > 1000)
        return (char *)"us";
    else
        return (char *)"ns";
}

static void
calculate_aggregate_metrics(struct gm_rdt_context_impl *ctx,
                            uint64_t current_time,
                            struct aggregate_metrics *metrics)
{
    uint64_t run_duration = current_time - ctx->metrics.start;

    memset(metrics, 0, sizeof(*metrics));
    if (!run_duration)
        return;

    double run_time_sec = run_duration / 1e9;

    /* time spent in main thread waiting for results from thread pool */
    double wait_percent = ((double)ctx->metrics.wait_duration / run_duration) * 100.0;
    /* time spent in main thread scheduling more work for thread pool */
    double scheduling_percent = 100.0 - wait_percent;

    uint64_t nodes_total = 0;
    uint64_t images_total = 0;
    uint64_t pixels_total = 0;
    uint64_t uvs_total = 0;
    uint64_t thresholds_total = 0;

    for (int i = 0; i < (int)ctx->thread_pool.size(); i++) {
        struct thread_state *state = &ctx->thread_pool[i];

        nodes_total += state->metrics.n_nodes;
        images_total += state->metrics.n_images_accumulated;
        pixels_total += state->metrics.n_pixels_accumulated;
        uvs_total += state->metrics.n_pixels_accumulated * ctx->n_uvs;
        thresholds_total += state->metrics.n_pixels_accumulated * ctx->n_thresholds;
    }

    double nodes_per_second = (double)nodes_total / run_time_sec;
    double images_per_sec = (double)images_total / run_time_sec;
    double px_per_sec = (double)pixels_total / run_time_sec;
    double uvs_per_sec = (double)uvs_total / run_time_sec;
    double thresholds_per_sec = (double)thresholds_total / run_time_sec;
    double pixels_per_node = (double)pixels_total / nodes_total;

    metrics->duration = run_duration;

    metrics->wait_percent = wait_percent;
    metrics->scheduling_percent = scheduling_percent;

    metrics->nodes_per_second = nodes_per_second;
    metrics->images_per_second = images_per_sec;
    metrics->pixels_per_second = px_per_sec;
    metrics->uvs_per_second = uvs_per_sec;
    metrics->thresholds_per_second = thresholds_per_sec;
    metrics->pixels_per_node = pixels_per_node;
}

static JSON_Value*
aggregate_metrics_to_json(struct gm_rdt_context_impl* ctx,
                          struct aggregate_metrics* metrics)
{
    JSON_Value *js = json_value_init_object();

    json_object_set_number(json_object(js), "duration", metrics->duration);

    json_object_set_number(json_object(js), "wait_percent", metrics->wait_percent);
    json_object_set_number(json_object(js), "scheduling_percent", metrics->scheduling_percent);

    json_object_set_number(json_object(js), "nodes_per_second", metrics->nodes_per_second);
    json_object_set_number(json_object(js), "images_per_second", metrics->images_per_second);
    json_object_set_number(json_object(js), "pixels_per_second", metrics->pixels_per_second);
    json_object_set_number(json_object(js), "uvs_per_second", metrics->uvs_per_second);
    json_object_set_number(json_object(js), "thresholds_per_second", metrics->thresholds_per_second);
    json_object_set_number(json_object(js), "pixels_per_node", metrics->pixels_per_node);

    return js;
}

static void
calculate_thread_metrics(struct gm_rdt_context_impl *ctx,
                         struct thread_state *state,
                         uint64_t current_time,
                         struct thread_metrics *metrics)
{
    uint64_t run_duration = current_time - state->metrics.start;

    memset(metrics, 0, sizeof(*metrics));
    if (!run_duration)
        return;

    double idle_percent = ((double)state->metrics.idle_duration /
        run_duration) * 100.0;
    double accu_percent = ((double)state->metrics.accumulation_time /
        run_duration) * 100.0;
    double rank_percent = ((double)state->metrics.gain_ranking_time /
        run_duration) * 100.0;

    double run_time_sec = run_duration / 1e9;
    double nodes_per_second = (double)state->metrics.n_nodes /
        run_time_sec;
    double images_per_sec = (double)state->metrics.n_images_accumulated /
        run_time_sec;
    double px_per_sec = (double)state->metrics.n_pixels_accumulated /
        run_time_sec;

    metrics->duration = run_duration;

    metrics->idle_percent = idle_percent;
    metrics->accumulation_percent = accu_percent;
    metrics->ranking_percent = rank_percent;

    metrics->nodes_per_second = nodes_per_second;
    metrics->images_per_second = images_per_sec;
    metrics->pixels_per_second = px_per_sec;
    metrics->uvs_per_second = px_per_sec * ctx->n_uvs;
    metrics->thresholds_per_second = px_per_sec * ctx->n_thresholds;
}

static JSON_Value*
thread_metrics_to_json(struct gm_rdt_context_impl* ctx,
                       struct thread_metrics* metrics)
{
    JSON_Value *js = json_value_init_object();

    json_object_set_number(json_object(js), "duration", metrics->duration);

    json_object_set_number(json_object(js), "idle_percent", metrics->idle_percent);
    json_object_set_number(json_object(js), "accumulation_percent", metrics->accumulation_percent);
    json_object_set_number(json_object(js), "ranking_percent", metrics->ranking_percent);

    json_object_set_number(json_object(js), "nodes_per_second", metrics->nodes_per_second);
    json_object_set_number(json_object(js), "images_per_second", metrics->images_per_second);
    json_object_set_number(json_object(js), "pixels_per_second", metrics->pixels_per_second);
    json_object_set_number(json_object(js), "uvs_per_second", metrics->uvs_per_second);
    json_object_set_number(json_object(js), "thresholds_per_second", metrics->thresholds_per_second);

    return js;
}

static void
log_thread_metrics(struct gm_rdt_context_impl *ctx,
                   int thread_index,
                   struct thread_metrics *metrics)
{

    gm_info(ctx->log, "%02d: over %.2f%s: idle %5.2f%%, acc %5.2f%% (%5.2f nd/s %6d img/s, %7d px/s, %7d uvs/s, %7d thresh/s), ranking %5.2f%%",
            thread_index,
            get_format_duration(metrics->duration),
            get_format_duration_suffix(metrics->duration),

            metrics->idle_percent,
            metrics->accumulation_percent,

            metrics->nodes_per_second,
            metrics->images_per_second,
            metrics->pixels_per_second,
            metrics->uvs_per_second,
            metrics->thresholds_per_second,

            metrics->ranking_percent
            );
}

static void
maybe_log_thread_metrics(struct gm_rdt_context_impl *ctx,
                         struct thread_state *state,
                         uint64_t current_time)
{
    if (current_time - state->last_metrics_log > 1000000000) {
        struct thread_metrics metrics;
        calculate_thread_metrics(ctx, state, current_time, &metrics);
        log_thread_metrics(ctx, state->idx, &metrics);
        state->last_metrics_log = current_time;
    }
}

static void
log_aggregate_metrics(struct gm_rdt_context_impl* ctx,
                      struct aggregate_metrics* metrics)
{
    gm_info(ctx->log, "aggregated over %.2f%s: sched %5.2f%%, wait work %5.2f%% (%5.2f nd/s %6d img/s, %7d px/s, %7d uvs/s, %7d thresh/s, %d px/nd)",
            get_format_duration(metrics->duration),
            get_format_duration_suffix(metrics->duration),

            metrics->scheduling_percent,
            metrics->wait_percent,

            metrics->nodes_per_second,
            metrics->images_per_second,
            metrics->pixels_per_second,
            metrics->uvs_per_second,
            metrics->thresholds_per_second,

            metrics->pixels_per_node
            );
}

static int
id_to_depth(int id)
{
    uint32_t id32 = id;

    /* clz() counts the number of leading zeros.
     *
     * 32-clz(value) essentially calculates an integer (rounded-down)
     * log2(value) (for a 32bit integer)
     *
     * Our IDs are generally base 0 (i.e. root node has ID = 0) but with base-1
     * IDs then every depth level neatly starts with a power of two ID (thus
     * the (id32 + 1))
     *
     * Finally subtract 1 to return a base-0 ID.
     */
    return 32 - __builtin_clz(id32 + 1) - 1;
}

/* For every image, pick N (ctx->n_pixels) random points within the silhoette
 * of the example pose for that frame.
 */
static Int3D*
generate_randomized_sample_points(struct gm_rdt_context_impl* ctx,
                                  int* total_count)
{
    //std::random_device rd;
    std::mt19937 rng(ctx->seed);
    //std::default_random_engine rng(ctx->seed);
    std::uniform_real_distribution<float> rand_0_1(0.0, 1.0);

    int n_image_pixels = ctx->width * ctx->height;
    std::vector<int> in_body_pixels(n_image_pixels);
    std::vector<int> indices(n_image_pixels);

    int n_pixels = ctx->n_images * ctx->n_pixels;
    Int3D* pixels = (Int3D*)xmalloc(n_pixels * sizeof(Int3D));

    for (int i = 0; i < ctx->n_images; i++)
    {
        int64_t image_idx = i * n_image_pixels;
        uint8_t* label_image = &ctx->label_images[image_idx];

        /* Our tracking system assumes that the body has been segmented
         * from the background before we try and label the different parts
         * of the body and so we're only interested in sampling points
         * inside the body...
         */
        in_body_pixels.clear();
        for (int y = 0; y < ctx->height; y++) {
            for (int x = 0; x < ctx->width; x++) {
                int off = y * ctx->width + x;
                int label = (int)label_image[off];

                gm_assert(ctx->log, label < ctx->n_labels,
                          "Label '%d' is bigger than expected (max %d)\n",
                          label, ctx->n_labels - 1);

                if (label != 0) { // 0 = background
                    in_body_pixels.push_back(off);
                }
            }
        }

        /* Note: we don't do anything to filter out duplicates which could
         * be fairly likely for frames where the body is relatively small.
         *
         * It seems best to not bias how many samples we consider across
         * the body based on the in-frame size, so our training expends
         * approximately the same amount of energy training on each pose
         * regardless of body size or distance from the camera.
         */
        int n_body_points = in_body_pixels.size();
        indices.clear();
        for (int j = 0; j < ctx->n_pixels; j++) {
            int off = rand_0_1(rng) * n_body_points;
            indices.push_back(off);
        }

        /* May slightly improve cache access patterns if we can process
         * our samples in memory order, even though the UV sampling
         * is somewhat randomized relative to these pixels...
         */
        std::sort(indices.begin(), indices.end());

        for (int j = 0; j < ctx->n_pixels; j++) {
            int off = in_body_pixels[indices[j]];

            int x = off % ctx->width;
            int y = off / ctx->width;

            Int3D pixel;
            pixel.xy[0] = x;
            pixel.xy[1] = y;
            pixel.i = i;

            pixels[i * ctx->n_pixels + j] = pixel;
        }
    }

    *total_count = n_pixels;
    return pixels;
}

static inline Int2D
normalize_histogram(int* histogram, int n_labels, float* normalized)
{
    Int2D sums = { 0, 0 };

    for (int i = 0; i < n_labels; i++)
    {
        if (histogram[i] > 0)
        {
            sums[0] += histogram[i];
            ++sums[1];
        }
    }

    if (sums[0] > 0)
    {
        for (int i = 0; i < n_labels; i++)
        {
            normalized[i] = histogram[i] / (float)sums[0];
        }
    }
    else
    {
        memset(normalized, 0, n_labels * sizeof(float));
    }

    return sums;
}

static inline float
calculate_shannon_entropy(float* normalized_histogram, int n_labels)
{
    float entropy = 0.f;
    for (int i = 0; i < n_labels; i++)
    {
        float value = normalized_histogram[i];
        if (value > 0.f && value < 1.f)
        {
            entropy += -value * log2f(value);
        }
    }
    return entropy;
}

static inline float
calculate_gain(float entropy, int n_pixels,
               float l_entropy, int l_n_pixels,
               float r_entropy, int r_n_pixels)
{
    return entropy - ((l_n_pixels / (float)n_pixels * l_entropy) +
                      (r_n_pixels / (float)n_pixels * r_entropy));
}

static void
accumulate_pixel_histograms(struct gm_rdt_context_impl* ctx,
                            NodeTrainData* data,
                            int* node_histogram)
{
    int p;
    int n_pixels = data->n_pixels;
    int n_labels = ctx->n_labels;
    int width = ctx->width;
    int height = ctx->height;

    for (p = 0; p < n_pixels; p++)
    {
        Int2D pixel = data->pixels[p].xy;
        int i = data->pixels[p].i;

        if (p % 10000 == 0) {
            if (interrupted)
                break;
        }

        int64_t image_idx = (int64_t)i * width * height;

        uint8_t* label_image = &ctx->label_images[image_idx];

        int pixel_idx = (pixel[1] * width) + pixel[0];
        int label = (int)label_image[pixel_idx];

        gm_assert(ctx->log, label < n_labels,
                  "Label '%d' is bigger than expected (max %d)\n",
                  label, n_labels - 1);

        ++node_histogram[label];
    }
}

static void
accumulate_uvt_lr_histograms(struct gm_rdt_context_impl* ctx,
                             struct thread_state *state,
                             NodeTrainData* data,
                             int uv_start, int uv_end,
                             int* node_histogram,
                             int* uvt_lr_histograms)
{
    int p;
    int last_i = -1;
    int max_depth = ctx->max_depth;
    int node_depth = id_to_depth(data->id);
    int n_pixels = data->n_pixels;
    int n_labels = ctx->n_labels;
    int n_thresholds = ctx->n_thresholds;
    int width = ctx->width;
    int height = ctx->height;

    for (p = 0; p < n_pixels; p++)
    {
        Int2D pixel = data->pixels[p].xy;
        int i = data->pixels[p].i;

        state->metrics.n_pixels_accumulated++;
        if (i != last_i) {
            state->metrics.n_images_accumulated++;
            last_i = i;
        }

        if (p % 10000 == 0) {
            if (interrupted)
                break;
            if (ctx->profile)
                maybe_log_thread_metrics(ctx, state, get_time());
        }

        int64_t image_idx = (int64_t)i * width * height;

        half* depth_image = &ctx->depth_images[image_idx];
        uint8_t* label_image = &ctx->label_images[image_idx];

        int pixel_idx = (pixel[1] * width) + pixel[0];
        int label = (int)label_image[pixel_idx];
        float depth = depth_image[pixel_idx];

        gm_assert(ctx->log, label < n_labels,
                  "Label '%d' is bigger than expected (max %d)\n",
                  label, n_labels - 1);

        ++node_histogram[label];

        // Don't waste processing time if this is the last depth
        if (node_depth >= max_depth - 1) {
            continue;
        }

        // Accumulate LR branch histograms

        // Sample pixels
        float samples[uv_end - uv_start];
        for (int c = uv_start; c < uv_end; c++)
        {
            UVPair uv = ctx->uvs[c];
            samples[c - uv_start] = sample_uv(depth_image,
                                              width, height,
                                              pixel, depth, uv);
        }

        // Partition on thresholds
        for (int c = 0, lr_histogram_idx = 0; c < uv_end - uv_start; c++)
        {
            for (int t = 0; t < n_thresholds;
                 t++, lr_histogram_idx += n_labels * 2)
            {
                // Accumulate histogram for this particular uvt combination
                // on both theoretical branches
                float threshold = ctx->thresholds[t];
                ++uvt_lr_histograms[samples[c] < threshold ?
                    lr_histogram_idx + label :
                    lr_histogram_idx + n_labels + label];
            }
        }
    }
}

static void*
worker_thread_cb(void* userdata)
{
    struct thread_state *state = (struct thread_state *)userdata;
    struct gm_rdt_context_impl* ctx = state->ctx;

    // Histogram for the node being processed
    int node_histogram[ctx->n_labels];

    // Histograms for each uvt combination being tested
    std::vector<int> uvt_lr_histograms;

    // We don't expect to be asked to process more than this many uvt
    // combos at a time so we can allocate the memory up front...
    int max_uvt_combos_per_thread = (ctx->n_uvs + ctx->n_threads/2) / ctx->n_threads;
    uvt_lr_histograms.reserve(max_uvt_combos_per_thread);

    while (1)
    {
        struct work work = {};
        struct result result = {};
        NodeTrainData *node_data = NULL;

        uint64_t idle_start = get_time();

        pthread_mutex_lock(&ctx->work_queue_lock);
        if (!ctx->work_queue.empty()) {
            work = ctx->work_queue.front();
            ctx->work_queue.pop();
        } else {
            while (!interrupted) {
                pthread_cond_wait(&ctx->work_queue_changed, &ctx->work_queue_lock);
                if (!ctx->work_queue.empty()) {
                    work = ctx->work_queue.front();
                    ctx->work_queue.pop();
                    break;
                }
            }
        }
        pthread_mutex_unlock(&ctx->work_queue_lock);

        uint64_t idle_end = get_time();
        state->metrics.idle_duration += (idle_end - idle_start);

        if (interrupted)
            break;

        gm_assert(ctx->log, work.node != NULL, "Spurious NULL work node");
        node_data = work.node;

        int node_depth = id_to_depth(node_data->id);

        // Clear histogram accumulators
        memset(node_histogram, 0, sizeof(node_histogram));
        uvt_lr_histograms.clear();
        uvt_lr_histograms.resize(ctx->n_labels *
                                 (work.uv_end - work.uv_start) *
                                 ctx->n_thresholds * 2);

        // Accumulate histograms
        uint64_t accu_start = get_time();
        accumulate_uvt_lr_histograms(ctx,
                                     state,
                                     node_data,
                                     work.uv_start, work.uv_end,
                                     node_histogram,
                                     uvt_lr_histograms.data());
        uint64_t accu_end = get_time();
        state->metrics.accumulation_time += accu_end - accu_start;

        // Calculate the normalised label histogram and get the number of pixels
        // and the number of labels in the root histogram.
        Int2D root_n_pixels = normalize_histogram(node_histogram,
                                                  ctx->n_labels,
                                                  result.nhistogram);

        // Determine the best u,v,t combination
        result.best_gain = 0.f;

        // If there's only 1 label, skip all this, gain is zero
        if (root_n_pixels[1] > 1 && node_depth < ctx->max_depth - 1)
        {
            uint64_t rank_start = get_time();

            // Calculate the shannon entropy for the normalised label histogram
            float entropy = calculate_shannon_entropy(result.nhistogram,
                                                      ctx->n_labels);

            // Calculate the gain for each combination of u,v,t and store the best
            for (int i = work.uv_start, lr_histo_base = 0;
                 i < work.uv_end && !interrupted; i++)
            {
                for (int j = 0; j < ctx->n_thresholds && !interrupted;
                     j++, lr_histo_base += ctx->n_labels * 2)
                {
                    float nhistogram[ctx->n_labels];
                    float l_entropy, r_entropy, gain;

                    Int2D l_n_pixels =
                        normalize_histogram(&uvt_lr_histograms[lr_histo_base],
                                            ctx->n_labels, nhistogram);
                    if (l_n_pixels[0] == 0 || l_n_pixels[0] == root_n_pixels[0])
                    {
                        continue;
                    }
                    l_entropy = calculate_shannon_entropy(nhistogram,
                                                          ctx->n_labels);

                    Int2D r_n_pixels =
                        normalize_histogram(
                            &uvt_lr_histograms[lr_histo_base + ctx->n_labels],
                            ctx->n_labels, nhistogram);
                    r_entropy = calculate_shannon_entropy(nhistogram,
                                                          ctx->n_labels);

                    gain = calculate_gain(entropy, root_n_pixels[0],
                                          l_entropy, l_n_pixels[0],
                                          r_entropy, r_n_pixels[0]);

                    if (gain > result.best_gain) {
                        result.best_gain = gain;
                        result.best_uv = i;
                        result.best_threshold = j;
                        result.n_lr_pixels[0] = l_n_pixels[0];
                        result.n_lr_pixels[1] = r_n_pixels[0];
                    }
                }
            }
            uint64_t rank_end = get_time();
            state->metrics.gain_ranking_time += rank_end - rank_start;
        }

        state->metrics.n_nodes++;
        uint64_t work_end = get_time();
        result.duration = work_end - idle_end;

        pthread_mutex_lock(&ctx->results_lock);
        ctx->results.push_back(result);
        pthread_cond_signal(&ctx->results_changed);
        pthread_mutex_unlock(&ctx->results_lock);
    }

    return NULL;
}

static void
collect_pixels(struct gm_rdt_context_impl* ctx,
               NodeTrainData* data,
               UVPair uv, float t,
               Int3D** l_pixels, Int3D** r_pixels, int* n_lr_pixels)
{
    *l_pixels = (Int3D*)xmalloc((n_lr_pixels[0] ? n_lr_pixels[0] :
                                 data->n_pixels) *
                                sizeof(Int3D));
    *r_pixels = (Int3D*)xmalloc((n_lr_pixels[1] ? n_lr_pixels[1] :
                                 data->n_pixels) *
                                sizeof(Int3D));

    int l_index = 0;
    int r_index = 0;
    for (int p = 0; p < data->n_pixels; p++)
    {
        Int3D* pixel = &data->pixels[p];
        int64_t image_idx = (int64_t)pixel->i * ctx->width * ctx->height;
        half* depth_image = &ctx->depth_images[image_idx];

        float depth = depth_image[(pixel->xy[1] * ctx->width) + pixel->xy[0]];
        float value = sample_uv(depth_image, ctx->width, ctx->height,
                                pixel->xy, depth, uv);

        if (value < t)
        {
            (*l_pixels)[l_index++] = *pixel;
        }
        else
        {
            (*r_pixels)[r_index++] = *pixel;
        }
    }

    if (n_lr_pixels[0] != l_index)
    {
        *l_pixels = (Int3D*)xrealloc(*l_pixels, l_index * sizeof(Int3D));
        n_lr_pixels[0] = l_index;
    }

    if (n_lr_pixels[1] != r_index)
    {
        *r_pixels = (Int3D*)xrealloc(*r_pixels, r_index * sizeof(Int3D));
        n_lr_pixels[1] = r_index;
    }
}

void
sigint_handler(int signum)
{
    interrupt_reason = "User interrupted";
    interrupted = true;
}

struct gm_ui_properties *
gm_rdt_context_get_ui_properties(struct gm_rdt_context *_ctx)
{
    struct gm_rdt_context_impl *ctx = (struct gm_rdt_context_impl *)_ctx;
    return &ctx->properties_state;
}

struct gm_rdt_context *
gm_rdt_context_new(struct gm_logger *log)
{
    struct gm_rdt_context_impl *ctx = new gm_rdt_context_impl();
    ctx->log = log;

    struct gm_ui_property prop;

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));

    ctx->data_dir = strdup(cwd);
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "data_dir";
    prop.desc = "Location of training data";
    prop.type = GM_PROPERTY_STRING;
    prop.string_state.ptr = &ctx->data_dir;
    ctx->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "index_name";
    prop.desc = "Name of frame index to load";
    prop.type = GM_PROPERTY_STRING;
    prop.string_state.ptr = &ctx->index_name;
    ctx->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "out_file";
    prop.desc = "Filename of tree to write";
    prop.type = GM_PROPERTY_STRING;
    prop.string_state.ptr = &ctx->out_filename;
    ctx->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "reload";
    prop.desc = "Filename of pre-existing tree to reload";
    prop.type = GM_PROPERTY_STRING;
    prop.string_state.ptr = &ctx->reload;
    ctx->properties.push_back(prop);

    ctx->n_pixels = 2000;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_pixels";
    prop.desc = "Number of pixels to sample per image";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_pixels;
    prop.int_state.min = 1;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->n_thresholds = 50;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_thresholds";
    prop.desc = "Number of thresholds to test";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_thresholds;
    prop.int_state.min = 1;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->threshold_range = 1.29;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "threshold_range";
    prop.desc = "Range of thresholds to test";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->threshold_range;
    prop.float_state.min = 0;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->n_uvs = 2000;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_uvs";
    prop.desc = "Number of UV combinations to test";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_uvs;
    prop.int_state.min = 1;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->uv_range = 1.29;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "uv_range";
    prop.desc = "Range of UV combinations to test";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->uv_range;
    prop.float_state.min = 0;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->max_depth = 20;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_depth";
    prop.desc = "Depth to train tree to";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->max_depth;
    // a value of 1 would just mean associating a random selection of pixels
    // with the root node and calculating the histograms of labels without
    // any decisions, so set the minimum to 2...
    prop.int_state.min = 2;
    prop.int_state.max = 30;
    ctx->properties.push_back(prop);

    ctx->max_nodes = 0;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_nodes";
    prop.desc = "Maximum number of nodes to train (for debug)";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->max_nodes;
    prop.int_state.min = 0;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->seed = 0;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "seed";
    prop.desc = "Seed to use for RNG";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->seed;
    prop.int_state.min = 0;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->pretty = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "pretty";
    prop.desc = "Pretty JSON output";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->pretty;
    ctx->properties.push_back(prop);

    ctx->verbose = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "verbose";
    prop.desc = "Verbose logging output";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->verbose;
    ctx->properties.push_back(prop);

    ctx->profile = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "profile";
    prop.desc = "Verbose profiling output";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->profile;
    ctx->properties.push_back(prop);

    ctx->n_threads = std::thread::hardware_concurrency();
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_threads";
    prop.desc = "Number of threads to spawn";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_threads;
    prop.int_state.min = 1;
    prop.int_state.max = 128;
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    return (struct gm_rdt_context *)ctx;
}

static void
destroy_training_state(struct gm_rdt_context_impl* ctx)
{
    xfree(ctx->uvs);
    ctx->uvs = NULL;
    xfree(ctx->thresholds);
    ctx->thresholds = NULL;
    xfree(ctx->label_images);
    ctx->label_images = NULL;
    xfree(ctx->depth_images);
    ctx->depth_images = NULL;
    if (ctx->history) {
        json_value_free(ctx->history);
        ctx->history = NULL;
    }
    if (ctx->record) {
        json_value_free(ctx->record);
        ctx->record = NULL;
    }
    if (ctx->data_meta) {
        json_value_free(ctx->data_meta);
        ctx->data_meta = NULL;
    }
}

void
gm_rdt_context_destroy(struct gm_rdt_context *_ctx)
{
    struct gm_rdt_context_impl *ctx = (struct gm_rdt_context_impl *)_ctx;
    destroy_training_state(ctx);
    delete ctx;
}

static JSON_Value*
recursive_build_tree(struct gm_rdt_context_impl* ctx,
                     Node* node,
                     int depth,
                     int id)
{
    JSON_Value* json_node_val = json_value_init_object();
    JSON_Object* json_node = json_object(json_node_val);

    if (node->label_pr_idx == 0)
    {
        json_object_set_number(json_node, "t", node->t);

        JSON_Value* u_val = json_value_init_array();
        JSON_Array* u = json_array(u_val);
        json_array_append_number(u, node->uv[0]);
        json_array_append_number(u, node->uv[1]);
        json_object_set_value(json_node, "u", u_val);

        JSON_Value* v_val = json_value_init_array();
        JSON_Array* v = json_array(v_val);
        json_array_append_number(v, node->uv[2]);
        json_array_append_number(v, node->uv[3]);
        json_object_set_value(json_node, "v", v_val);

        if (depth < (ctx->max_depth - 1))
        {
            /* NB: The nodes in .rdt files are in a packed array arranged in
             * breadth-first, left then right child order with the root node at
             * index zero.
             *
             * With this layout then given an index for any particular node
             * ('id' here) then 2 * id + 1 is the index for the left child and
             * 2 * id + 2 is the index for the right child...
             */
            int left_id = id * 2 + 1;
            Node* left_node = &ctx->tree[left_id];
            int right_id = id * 2 + 2;
            Node* right_node = &ctx->tree[right_id];

            JSON_Value* left_json = recursive_build_tree(ctx, left_node,
                                                         depth + 1, left_id);
            json_object_set_value(json_node, "l", left_json);
            JSON_Value* right_json = recursive_build_tree(ctx, right_node,
                                                          depth + 1, right_id);
            json_object_set_value(json_node, "r", right_json);
        }
    }
    else if (node->label_pr_idx != INT_MAX) // Return empty obj for untrained nodes
    {
        JSON_Value* probs_val = json_value_init_array();
        JSON_Array* probs = json_array(probs_val);

        /* NB: node->label_pr_idx is a base-one index since index zero is
         * reserved to indicate that the node is not a leaf node
         */
        float* pr_table = &ctx->tree_histograms[(node->label_pr_idx - 1) *
            ctx->n_labels];

        for (int i = 0; i < ctx->n_labels; i++)
        {
            json_array_append_number(probs, pr_table[i]);
        }

        json_object_set_value(json_node, "p", probs_val);
    }

    return json_node_val;
}

/* TODO: include more information than the RDTree, such as a date timestamp,
 * and all our hyperparameters such as: n_thresholds, threshold_range, n_pixels
 * etc
 */
static bool
save_tree_json(struct gm_rdt_context_impl *ctx,
               std::vector<Node> &tree,
               std::vector<float> &tree_histograms,
               const char* filename)
{
    JSON_Value *rdt = json_value_init_object();

    JSON_Value *record_val = ctx->record;
    JSON_Value *history = json_value_deep_copy(ctx->history);
    if (!history) {
        history = json_value_init_array();
    }
    json_array_append_value(json_array(history), record_val);

    json_object_set_value(json_object(rdt), "history", history);

    json_object_set_number(json_object(rdt), "depth", ctx->max_depth);
    json_object_set_number(json_object(rdt), "vertical_fov", ctx->fov);

    /* TODO: we could remove the background label
     *
     * Since we no longer train to be able to classify the background we
     * currently waste space in the leaf-node probability tables with an unused
     * slot associated with this bg_label.
     *
     * At least one thing to consider with removing this is that infer.cc
     * would need adjusting to perhaps reserve an extra slot in its returned
     * probabilities for a background label, or perhaps assume the caller
     * knows to ignore background pixels.
     */
    json_object_set_number(json_object(rdt), "n_labels", ctx->n_labels);
    json_object_set_number(json_object(rdt), "bg_label", 0);

    JSON_Value* labels = json_object_get_value(json_object(ctx->data_meta),
                                               "labels");
    labels = json_value_deep_copy(labels);
    json_object_set_value(json_object(rdt), "labels", labels);

    JSON_Value *nodes = recursive_build_tree(ctx, &tree[0], 0, 0);

    json_object_set_value(json_object(rdt), "root", nodes);

    JSON_Status status;
    if (ctx->pretty)
        status = json_serialize_to_file_pretty(rdt, filename);
    else
        status = json_serialize_to_file(rdt, filename);
    if (status != JSONSuccess)
    {
        fprintf(stderr, "Failed to serialize output to JSON\n");
        return false;
    }

    return true;
}

static bool
reload_tree(struct gm_rdt_context_impl* ctx,
            const char* filename,
            NodeTrainData &root_node,
            char** err)
{
    gm_info(ctx->log, "Reloading %s...\n", filename);

    int len = strlen(filename);
    RDTree* checkpoint = NULL;
    if (len > 5 && strcmp(filename + len - 5, ".json") == 0) {
        JSON_Value* js = json_parse_file(filename);
        if (!js) {
            gm_throw(ctx->log, err, "Failed to parse %s", filename);
            return false;
        }

        checkpoint = rdt_tree_load_from_json(ctx->log, js, err);

        JSON_Value* history = json_object_get_value(json_object(js), "history");
        if (history) {
            ctx->history = json_value_deep_copy(history);
        }

        JSON_Value* labels = json_object_get_value(json_object(ctx->data_meta), "labels");
        JSON_Value* rlabels = json_object_get_value(json_object(js), "labels");
        if (labels) {
            JSON_Array* labels_array = json_array(labels);
            JSON_Array* rlabels_array = json_array(rlabels);
            int n_labels = json_array_get_count(labels_array);
            int n_rlabels = json_array_get_count(rlabels_array);
            if (n_labels != n_rlabels || n_labels != ctx->n_labels) {
                gm_throw(ctx->log, err, "%s has %d labels, expected %d\n",
                         filename,
                         n_rlabels, ctx->n_labels);
                return false;
            }
            for (int i = 0; i < n_labels; i++) {
                JSON_Object* label = json_array_get_object(labels_array, i);
                JSON_Object* rlabel = json_array_get_object(rlabels_array, i);
                if (strcmp(json_object_get_string(label, "name"),
                           json_object_get_string(rlabel, "name")) != 0)
                {
                    gm_throw(ctx->log, err, "%s label semantics don't match those of the training data",
                             filename);
                    return false;
                }
            }
        }

        json_value_free(js);
    } else {
        checkpoint = rdt_tree_load_from_file(ctx->log, filename, err);
    }
    if (!checkpoint)
        return false;

    // Do some basic validation
    if (checkpoint->header.n_labels != ctx->n_labels)
    {
        gm_throw(ctx->log, err, "%s has %d labels, expected %d\n",
                 filename,
                 (int)checkpoint->header.n_labels, ctx->n_labels);
        return false;
    }

    if (fabs(checkpoint->header.fov - ctx->fov) > 1e-6)
    {
        gm_throw(ctx->log, err, "%s has FOV %.2f, expected %.2f\n",
                 filename,
                 checkpoint->header.fov, ctx->fov);
        return false;
    }

    if (ctx->max_depth < checkpoint->header.depth)
        gm_warn(ctx->log, "Pruning tree with more levels that needed");

    int reload_depth = std::min((int)checkpoint->header.depth, ctx->max_depth);
    gm_info(ctx->log, "Reloading %d levels", reload_depth);

    // Restore nodes
    int n_reload_nodes = (1<<reload_depth) - 1;
    gm_info(ctx->log, "Reloading %d nodes", n_reload_nodes);
    memcpy(ctx->tree.data(), checkpoint->nodes, n_reload_nodes * sizeof(Node));

    // Navigate the tree to determine any unfinished nodes and the last
    // trained depth
    std::queue<NodeTrainData> reload_queue;

    reload_queue.push(root_node);

    while (reload_queue.size())
    {
        NodeTrainData node_data = reload_queue.front();
        int node_depth = id_to_depth(node_data.id);
        reload_queue.pop();
        Node* node = &ctx->tree[node_data.id];

        if (node->label_pr_idx == INT_MAX)
        {
            // INT_MAX implies it wasn't trained yet..
            ctx->train_queue.push(node_data);
        }
        else if (node->label_pr_idx != 0 && // leaf node
                 node_depth == (checkpoint->header.depth - 1) &&
                 ctx->max_depth > checkpoint->header.depth)
        {
            /* Also need to train leaf nodes in the last level of the reloaded
             * tree if we're training deeper now...
             */
            node->label_pr_idx = INT_MAX;
            ctx->train_queue.push(node_data);
        }
        else if (node_depth == (ctx->max_depth - 1) &&
                 node->label_pr_idx == 0 && // wasn't previously a leaf node
                 checkpoint->header.depth > ctx->max_depth)
        {
            /* If we're pruning an existing tree to a shallower depth then
             * all nodes on the last level that weren't already leaf nodes
             * need to be re-trained to calculate histograms...
             */
            node->label_pr_idx = INT_MAX;
            ctx->train_queue.push(node_data);
        }
        else
        {
            // Leaf nodes are associated with a histogram which we need to
            // copy across to tree_histograms...
            if (node->label_pr_idx != 0)
            {
                float* pr_table = &checkpoint->
                    label_pr_tables[ctx->n_labels * (node->label_pr_idx - 1)];

                /* Note: we make no assumption about the ordering of histograms
                 * in the loaded tree, but the indices aren't necessarily
                 * preserved as we copy them across to tree_histograms...
                 */
                int len = ctx->tree_histograms.size();
                ctx->tree_histograms.resize(len + ctx->n_labels);
                memcpy(&ctx->tree_histograms[len], pr_table, ctx->n_labels * sizeof(float));

                // NB: 0 is reserved for non-leaf nodes
                node->label_pr_idx = (len / ctx->n_labels) + 1;
            } else {
                // If the node isn't a leaf-node, calculate which pixels should
                // go to the next two nodes and add them to the reload
                // queue
                Int3D* l_pixels;
                Int3D* r_pixels;
                int n_lr_pixels[] = { 0, 0 };
                collect_pixels(ctx, &node_data,
                               node->uv, node->t,
                               &l_pixels, &r_pixels,
                               n_lr_pixels);

                int id = (2 * node_data.id) + 1;

                NodeTrainData ldata;
                ldata.id = id;
                ldata.n_pixels = n_lr_pixels[0];
                ldata.pixels = l_pixels;

                NodeTrainData rdata;
                rdata.id = id + 1;
                rdata.n_pixels = n_lr_pixels[1];
                rdata.pixels = r_pixels;

                reload_queue.push(ldata);
                reload_queue.push(rdata);
            }

            // Since we didn't add the node to the training queue we
            // no longer need the associated pixel data for this node...
            xfree(node_data.pixels);
        }
    }

    rdt_tree_destroy(checkpoint);

    if (!ctx->train_queue.size())
    {
        gm_throw(ctx->log, err, "Tree already fully trained.\n");
        return false;
    }

    return true;
}

static JSON_Value*
create_training_record(struct gm_rdt_context_impl* ctx)
{
    JSON_Value *record_val = json_value_init_object();

    time_t unix_time = time(NULL);
    struct tm cur_time = *localtime(&unix_time);
    asctime(&cur_time);

    char date_str[256];
    if (snprintf(date_str, sizeof(date_str), "%d-%d-%d-%d-%d-%d",
                 (int)cur_time.tm_year + 1900,
                 (int)cur_time.tm_mon + 1,
                 (int)cur_time.tm_mday,
                 (int)cur_time.tm_hour,
                 (int)cur_time.tm_min,
                 (int)cur_time.tm_sec) >= (int)sizeof(date_str))
    {
        gm_error(ctx->log, "Unable to format date string");
    } else {
        json_object_set_string(json_object(record_val), "date", date_str);
    }

    gm_props_to_json(ctx->log, &ctx->properties_state, record_val);

    return record_val;
}

static void
print_label_histogram(struct gm_logger* log,
                      JSON_Array* labels,
                      float* histogram,
                      int histogram_len)
{
    static const char *bars[] = {
        " ",
        "▏",
        "▎",
        "▍",
        "▌",
        "▋",
        "▊",
        "▉",
        "█"
    };

    int max_bar_width = 30; // measured in terminal columns

    for (int i = 0; i < (int)json_array_get_count(labels); i++) {
        JSON_Object* label = json_array_get_object(labels, i);
        const char *name = json_object_get_string(label, "name");
        int bar_len = max_bar_width * 8 * histogram[i];
        char bar[max_bar_width * 4]; // room for multi-byte utf8 characters
        int bar_pos = 0;

        for (int j = 0; j < max_bar_width; j++) {
            int part;
            if (bar_len > 8) {
                part = 8;
                bar_len -= 8;
            } else {
                part = bar_len;
                bar_len = 0;
            }
            int len = strlen(bars[part]);
            memcpy(bar + bar_pos, bars[part], len);
            bar_pos += len;
        }
        bar[bar_pos++] = '\0';
        gm_info(log, "%-20s, %-3.0f%%|%s|", name, histogram[i] * 100.0f, bar);
    }
}

/* A histogram of the labels for the root node pixels is useful to help double
 * check they roughly match the relative sizes of the different labels else
 * maybe there was a problem with generating our sample points.
 */
static void
check_root_pixels_histogram(struct gm_rdt_context_impl* ctx,
                            NodeTrainData* root_node)
{
    gm_info(ctx->log, "Calculating root node pixel histogram");
    ctx->root_pixel_histogram.resize(ctx->n_labels);
    ctx->root_pixel_nhistogram.resize(ctx->n_labels);
    accumulate_pixel_histograms(ctx, root_node, ctx->root_pixel_histogram.data());
    normalize_histogram(ctx->root_pixel_histogram.data(),
                        ctx->n_labels,
                        ctx->root_pixel_nhistogram.data());
    JSON_Array* labels = json_object_get_array(json_object(ctx->data_meta),
                                               "labels");
    gm_info(ctx->log, "Histogram of root node pixel labels:");
    print_label_histogram(ctx->log,
                          labels,
                          ctx->root_pixel_nhistogram.data(),
                          ctx->root_pixel_nhistogram.size());

    JSON_Value* hist_val = json_value_init_array();
    JSON_Array* hist = json_array(hist_val);
    for (int i = 0; i < ctx->n_labels; i++) {
        json_array_append_number(hist, ctx->root_pixel_nhistogram[i]);
    }

    json_object_set_value(json_object(ctx->record), "root_pixels_histogram",
                          hist_val);
}

bool
gm_rdt_context_train(struct gm_rdt_context* _ctx, char** err)
{
    struct gm_rdt_context_impl* ctx = (struct gm_rdt_context_impl*)_ctx;
    TimeForDisplay since_begin, since_last;
    struct timespec begin, last, now;
    int n_threads = ctx->n_threads;

    /* Reset global state, in case a previous training run was interrupted... */
    interrupted = false;
    interrupt_reason = NULL;

    const char* data_dir = ctx->data_dir;
    if (!data_dir) {
        gm_throw(ctx->log, err, "Data directory not specified");
        return false;
    }
    const char* index_name = ctx->index_name;
    if (!index_name) {
        gm_throw(ctx->log, err, "Index name not specified");
        return false;
    }
    const char* out_filename = ctx->out_filename;
    if (!out_filename) {
        gm_throw(ctx->log, err, "Output filename not specified");
        return false;
    }

    ctx->record = create_training_record(ctx);

    gm_info(ctx->log, "Scanning training directories...\n");
    ctx->data_meta =
        gather_train_data(ctx->log,
                          data_dir,
                          index_name,
                          NULL, // no joint map
                          &ctx->n_images,
                          NULL, // ignore n_joints
                          &ctx->width, &ctx->height,
                          &ctx->depth_images, &ctx->label_images, NULL,
                          err);
    if (!ctx->data_meta) {
        destroy_training_state(ctx);
        return false;
    }

    JSON_Object* meta_camera =
        json_object_get_object(json_object(ctx->data_meta), "camera");
    ctx->fov = json_object_get_number(meta_camera, "vertical_fov");

    ctx->n_labels = json_object_get_number(json_object(ctx->data_meta), "n_labels");

    gm_assert(ctx->log, ctx->n_labels <= MAX_LABELS,
              "Can't handle training with more than %d labels",
              MAX_LABELS);
    gm_assert(ctx->log, (uint64_t)ctx->n_pixels * ctx->n_images < INT_MAX,
              "Can't handle training with more than %d pixels, but n_pixels * n_images = %" PRIu64,
              INT_MAX, (uint64_t)ctx->n_pixels * ctx->n_images);

    // Work out pixels per meter and adjust uv range accordingly
    float ppm = (ctx->height / 2.f) / tanf(ctx->fov / 2.f);
    ctx->uv_range *= ppm;

    // Calculate the u,v,t parameters that we're going to test
    gm_info(ctx->log, "Preparing training metadata...\n");
    ctx->uvs = (UVPair*)xmalloc(ctx->n_uvs * sizeof(UVPair));
    //std::random_device rd;
    std::mt19937 rng(ctx->seed);
    std::uniform_real_distribution<float> rand_uv(-ctx->uv_range / 2.f,
                                                  ctx->uv_range / 2.f);
    for (int i = 0; i < ctx->n_uvs; i++) {
        ctx->uvs[i][0] = rand_uv(rng);
        ctx->uvs[i][1] = rand_uv(rng);
        ctx->uvs[i][2] = rand_uv(rng);
        ctx->uvs[i][3] = rand_uv(rng);
    }
    ctx->thresholds = (float*)xmalloc(ctx->n_thresholds * sizeof(float));
    for (int i = 0; i < ctx->n_thresholds; i++) {
        ctx->thresholds[i] = -ctx->threshold_range / 2.f +
            (i * ctx->threshold_range / (float)(ctx->n_thresholds - 1));
    }

    // Allocate memory to store the decision tree.
    int n_tree_nodes = (1<<ctx->max_depth) - 1;
    ctx->tree.resize(n_tree_nodes);

    // Mark nodes in tree as unfinished, for checkpoint restoration
    // Note: we still do this if we are reloading a tree, since it may
    // be shallower than the total tree size.
    for (int i = 0; i < (int)ctx->tree.size(); i++) {
        ctx->tree[i].label_pr_idx = INT_MAX;
    }

    // Create the randomized sample points across all images that the decision
    // tree is going to learn to classify, and associate with a root node...
    //
    // The training recursively splits the pixels at each node of the tree,
    // either terminating when a branch runs out of pixels to differentiate
    // or after reaching the maximum training depth.
    //
    NodeTrainData root_node;
    root_node.id = 0;
    root_node.pixels = generate_randomized_sample_points(ctx, &root_node.n_pixels);

    check_root_pixels_histogram(ctx, &root_node);

    if (ctx->reload) {
        if (!reload_tree(ctx, ctx->reload, root_node, err)) {
            destroy_training_state(ctx);
            return false;
        }
    } else {
        ctx->train_queue.push(root_node);
    }

    ctx->per_depth_metrics.resize(ctx->max_depth);

    gm_info(ctx->log, "Initialising %u threads...\n", n_threads);
    ctx->thread_pool.resize(n_threads);
    for (int i = 0; i < n_threads; i++)
    {
        struct thread_state *state = &ctx->thread_pool[i];
        state->idx = i;
        state->ctx = ctx;
        state->per_depth_metrics.resize(ctx->max_depth);

        if (pthread_create(&state->thread, NULL,
                           worker_thread_cb, (void*)state) != 0)
        {
            gm_throw(ctx->log, err, "Error creating thread\n");
            destroy_training_state(ctx);
            return false;
        }
    }

    gm_info(ctx->log, "Beginning training...\n");
    signal(SIGINT, sigint_handler);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    last = begin;
    uint64_t last_metrics = get_time();
    int last_depth = -1;
    uint64_t wait_start, wait_end;
    while (ctx->train_queue.size())
    {
        int best_uv = 0;
        int best_threshold = 0;
        int *n_lr_pixels = NULL;
        float best_gain = 0.0;

        NodeTrainData node_data = ctx->train_queue.front();
        ctx->train_queue.pop();

        int node_depth = id_to_depth(node_data.id);

        if (node_depth != last_depth)
        {
            clock_gettime(CLOCK_MONOTONIC, &now);
            since_begin = get_time_for_display(&begin, &now);
            since_last = get_time_for_display(&last, &now);
            last = now;

            uint64_t current = get_time();

            if (last_depth != -1) { // Finished last_depth
                gm_info(ctx->log, "Finished level %d: ", last_depth + 1);
                calculate_aggregate_metrics(ctx, current,
                                            &ctx->per_depth_metrics[last_depth]);
                log_aggregate_metrics(ctx, &ctx->per_depth_metrics[last_depth]);

                for (int i = 0; i < ctx->n_threads; i++) {
                    struct thread_state *state = &ctx->thread_pool[i];

                    calculate_thread_metrics(ctx, state, current,
                                             &state->per_depth_metrics[last_depth]);
                    log_thread_metrics(ctx, state->idx, &state->per_depth_metrics[last_depth]);
                }
            }

            gm_info(ctx->log,
                    "(%02d:%02d:%02d / %02d:%02d:%02d) Training level %d (%d nodes)\n",
                    since_begin.hours, since_begin.minutes, since_begin.seconds,
                    since_last.hours, since_last.minutes, since_last.seconds,
                    node_depth + 1, (int)ctx->train_queue.size() + 1);

            for (int i = 0; i < ctx->n_threads; i++) {
                struct thread_state *state = &ctx->thread_pool[i];
                memset(&state->metrics, 0, sizeof(state->metrics));
                state->metrics.start = current;
            }
            memset(&ctx->metrics, 0, sizeof(ctx->metrics));
            ctx->metrics.start = current;

            last_depth = node_depth;
        }

        /* XXX: note we don't need to take the work_queue_lock here since the
         * current scheduling model implies there's no work being processed
         * by any of the threads at this point...
         */

        ctx->results.clear();
        int n_pending_results = 0;

        int n_uvs_per_work = ctx->n_uvs / n_threads;
        for (int i = 0; i < n_threads; i++) {
            struct work work;
            work.node = &node_data;
            work.uv_start = i * n_uvs_per_work;
            work.uv_end = (i == n_threads - 1) ? ctx->n_uvs : (i + 1) * n_uvs_per_work;
            ctx->work_queue.push(work);
            n_pending_results++;
        }

        wait_start = get_time();
        pthread_cond_broadcast(&ctx->work_queue_changed);

        pthread_mutex_lock(&ctx->results_lock);
        while (!interrupted && (int)ctx->results.size() < n_pending_results)
            pthread_cond_wait(&ctx->results_changed, &ctx->results_lock);
        pthread_mutex_unlock(&ctx->results_lock);
        wait_end = get_time();
        ctx->metrics.wait_duration += wait_end - wait_start;

        // Quit if we've been interrupted
        if (interrupted) {
            break;
        }

        if (ctx->profile) {
            uint64_t current = get_time();
            if (current - last_metrics > 1000000000) {
                for (int i = 0; i < ctx->n_threads; i++) {
                    struct thread_state *state = &ctx->thread_pool[i];
                    maybe_log_thread_metrics(ctx, state, current);
                }
                struct aggregate_metrics agg_metrics;
                calculate_aggregate_metrics(ctx, current, &agg_metrics);
                log_aggregate_metrics(ctx, &agg_metrics);
                last_metrics = current;
            }
        }

        // See which thread got the best uvt combination
        for (int i = 0; i < (int)ctx->results.size(); i++) {
            struct result *result = &ctx->results[i];
            if (result->best_gain > best_gain)
            {
                best_gain = result->best_gain;
                best_uv = result->best_uv;
                best_threshold = result->best_threshold;
                n_lr_pixels = result->n_lr_pixels;
            }
        }

        // Add this node to the tree and possibly add left/ride nodes to the
        // training queue.
        Node* node = &ctx->tree[node_data.id];
        if (best_gain > 0.f && (node_depth + 1) < ctx->max_depth)
        {
            node->uv = ctx->uvs[best_uv];
            node->t = ctx->thresholds[best_threshold];
            if (ctx->verbose)
            {
                gm_info(ctx->log,
                        "  Node (%u)\n"
                        "    Gain: %f\n"
                        "    U: (%f, %f)\n"
                        "    V: (%f, %f)\n"
                        "    T: %f\n",
                        node_data.id, best_gain,
                        node->uv[0], node->uv[1],
                        node->uv[2], node->uv[3],
                        node->t);
            }

            Int3D* l_pixels;
            Int3D* r_pixels;

            collect_pixels(ctx, &node_data, node->uv, node->t,
                           &l_pixels, &r_pixels, n_lr_pixels);

            int id = (2 * node_data.id) + 1;
            NodeTrainData ldata;
            ldata.id = id;
            ldata.n_pixels = n_lr_pixels[0];
            ldata.pixels = l_pixels;

            NodeTrainData rdata;
            rdata.id = id + 1;
            rdata.n_pixels = n_lr_pixels[1];
            rdata.pixels = r_pixels;

            ctx->train_queue.push(ldata);
            ctx->train_queue.push(rdata);

            // Mark the node as a continuing node
            node->label_pr_idx = 0;
        }
        else
        {
            /* Each result will include a normalized histogram of pixel labels
             * for the last node that was processed. They should all be
             * identical so we just refer to the first result...
             */
            float *nhistogram = ctx->results[0].nhistogram;

            if (ctx->verbose)
            {
                gm_info(ctx->log, "  Leaf node (%d)\n", node_data.id);
                for (int i = 0; i < ctx->n_labels; i++) {
                    if (nhistogram[i] > 0.f) {
                        gm_info(ctx->log, "    %02d - %f\n", i, nhistogram[i]);
                    }
                }
            }

            // NB: 0 is reserved for non-leaf nodes
            node->label_pr_idx = (ctx->tree_histograms.size() / ctx->n_labels) + 1;
            int len = ctx->tree_histograms.size();
            ctx->tree_histograms.resize(len + ctx->n_labels);
            memcpy(&ctx->tree_histograms[len], nhistogram, ctx->n_labels * sizeof(float));
        }

        // We no longer need the node's pixel data
        xfree(node_data.pixels);

        ctx->n_nodes_trained++;
        if (ctx->max_nodes && ctx->n_nodes_trained > ctx->max_nodes) {
            if (ctx->verbose)
                gm_warn(ctx->log, "Interrupting - Maximum number of nodes (%d) reached",
                        ctx->max_nodes);
            interrupt_reason = "Max nodes trained";
            interrupted = true;
        }
    }

    // Signal threads to free memory and quit
    //
    // Note: we need to take work_queue_lock otherwise there's a race within
    // the worker thread in the loop between checking the interrupted state and
    // starting pthread_cond_wait() where we will get a deadlock if we happen
    // to set interrupted = true at that point.
    pthread_mutex_lock(&ctx->work_queue_lock);
    interrupted = true;
    pthread_cond_broadcast(&ctx->work_queue_changed);
    pthread_mutex_unlock(&ctx->work_queue_lock);
    for (int i = 0; i < n_threads; i++)
    {
        struct thread_state *state = &ctx->thread_pool[i];

        if (pthread_join(state->thread, NULL) != 0) {
            gm_error(ctx->log, "Error joining thread, trying to continue...\n");
        }
    }

    JSON_Value *js_metrics = json_value_init_object();
    json_object_set_value(json_object(ctx->record), "metrics", js_metrics);
    JSON_Value *js_dmetrics = NULL; // per-depth metrics

    for (int i = 0; i < ctx->max_depth; i++) {
        struct aggregate_metrics *dmetrics = &ctx->per_depth_metrics[i];

        /* If we reloaded an existing tree then we might not have metrics for
         * every depth...
         */
        if (dmetrics->duration == 0)
            continue;

        if (!js_dmetrics) {
            js_dmetrics = json_value_init_array();
            json_object_set_value(json_object(js_metrics), "per_depth", js_dmetrics);
        }

        JSON_Value *js_depth = json_value_init_object();
        json_object_set_number(json_object(js_depth), "depth", i);
        json_array_append_value(json_array(js_dmetrics), js_depth);

        JSON_Value *js_agg_metrics = aggregate_metrics_to_json(ctx, dmetrics);
        json_object_set_value(json_object(js_depth), "aggregate", js_agg_metrics);

        JSON_Value *js_per_thread = json_value_init_array();
        json_object_set_value(json_object(js_depth), "per_thread", js_per_thread);

        for (int t = 0; t < n_threads; t++) {
            struct thread_state *state = &ctx->thread_pool[t];
            struct thread_metrics *tmetrics = &state->per_depth_metrics[i];

            if (tmetrics->duration != dmetrics->duration) {
                gm_warn(ctx->log, "Inconsistent duration between aggregate metrics and thread %d metrics for depth = %d",
                        state->idx, i);
            }

            JSON_Value *js_tmetrics = thread_metrics_to_json(ctx, &state->per_depth_metrics[i]);
            json_array_append_value(json_array(js_per_thread), js_tmetrics);
        }
    }

    // Write to file
    clock_gettime(CLOCK_MONOTONIC, &now);
    since_begin = get_time_for_display(&begin, &now);
    since_last = get_time_for_display(&last, &now);
    last = now;
    gm_info(ctx->log,
            "(%02d:%02d:%02d / %02d:%02d:%02d) Writing output to '%s'...\n",
            since_begin.hours, since_begin.minutes, since_begin.seconds,
            since_last.hours, since_last.minutes, since_last.seconds,
            out_filename);

    save_tree_json(ctx,
                   ctx->tree,
                   ctx->tree_histograms,
                   out_filename);

    clock_gettime(CLOCK_MONOTONIC, &now);
    since_begin = get_time_for_display(&begin, &now);
    since_last = get_time_for_display(&last, &now);
    last = now;
    gm_info(ctx->log,
            "(%02d:%02d:%02d / %02d:%02d:%02d) %s\n",
            since_begin.hours, since_begin.minutes, since_begin.seconds,
            since_last.hours, since_last.minutes, since_last.seconds,
            interrupt_reason ?: "Done!");

    // Free memory that isn't needed anymore
    destroy_training_state(ctx);

    return true;
}
