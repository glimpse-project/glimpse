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

#include <png.h>

#include "half.hpp"

#include "xalloc.h"
#include "rdt_tree.h"
#include "train_utils.h"
#include "image_utils.h"

#include "glimpse_rdt.h"
#include "glimpse_log.h"
#include "glimpse_properties.h"

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "rdt"

using half_float::half;


#define HUGE_DEPTH       1000.0

#define xsnprintf(dest, size, fmt, ...) do { \
        if (snprintf(dest, size, fmt,  __VA_ARGS__) >= (int)size) \
            exit(1); \
    } while(0)

#define nth_threshold(n, step) ({ \
            int offset_n = n + 1; /* we only want zero once */ \
            int sign = -1 + (n%2) * 2; \
            int threshold = offset_n/2 * sign * step; \
            threshold; /* statement expression evaluates to this */ \
        })

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

struct bounds {
    int min_x;
    int max_x;
    int min_y;
    int max_y;
};

struct depth_meta {
    int width;
    int height;
    int64_t pixel_offset;
};

struct pixel {
    int16_t x;
    int16_t y;
    uint32_t label:8;
    uint32_t i:24;
};

struct node_data {
    int id; // Unique id to place the node in a tree.
    int n_pixels; // Number of pixels that have reached this node.
    struct pixel* pixels;   // An array of pixel pairs and image indices.
};

/* Work submitted for the thread pool to process... */
struct work {
    int depth; // All work is associated with a depth to help collect
               // and report metrics

    void (*work_cb)(struct thread_state* state,
                    void* user_data);
    void* user_data;
};

#define MAX_LABELS 40
struct node_shard_data {
    bool done;
    float best_gain;
    int best_uv;
    int best_threshold;
    int n_lr_pixels[2];
    uint64_t duration;
};

struct node {
    int32_t uvs[4]; // U in [0:2] and V in [2:4]
    int32_t t_mm;
    uint32_t label_pr_idx;  // Index into label probability table (1-based)
};

#if 0
enum {
    RESULS_BUSY, // Submitted to be processed by a worker thread
    RESULT_DONE, // Worker thread has finished computing result
    RESULT_PROCESSING // Worker thread is processing / combining result with others
};
#endif

struct node_shard_results {
    int ref;
    pthread_mutex_t check_lock; // Use to avoid processing results more than
                                // once. Take lock and enumerate each data entry
                                // to see that they are all complete.
    bool taken;

    int n_node_labels; // How many labels have been observed for this node's pixels
    float nhistogram[MAX_LABELS];

    int n_shards;
    struct node_shard_data data[];
};

/* Instructions to process a subset of a single node along with a place to
 * store results. It is the collective responsibility of these workers to
 * queue a follow up worker to process the results after completing
 * a shard.
 */
struct node_shard_work {
    struct node_data node_data;
    int uv_start;
    int uv_end;
    struct node_shard_results* results;
    int shard_index;
};

struct process_node_shards_work {
    struct node_data node_data;
    struct node_shard_results* results;
};

struct thread_depth_metrics_report {
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

struct thread_depth_metrics_raw {
    // Mutually exclusive
    uint64_t idle_time;
    uint64_t work_time;

    uint64_t accumulation_time;
    uint64_t gain_ranking_time;

    uint64_t n_thresholds_accumulated;
    uint64_t n_uvs_accumulated;
    uint64_t n_pixels_accumulated;
    uint64_t n_images_accumulated;
    uint64_t n_nodes;
};

#define MAX_DEPTH 30

struct thread_state {
    struct gm_rdt_context_impl* ctx;
    int idx;

    pthread_t thread;

    /* We aim to use 16bit histograms when there are fewer than UINT16_MAX
     * pixels for the current node, since the write bandwidth to these
     * histograms can be a performance bottleneck
     */
    std::vector<uint32_t> uvt_lr_histograms_32;
    std::vector<uint16_t> uvt_lr_histograms_16;

    uint64_t current_work_start;
    uint64_t last_metrics_log;

    std::vector<thread_depth_metrics_raw> per_depth_metrics;
};

struct gm_rdt_context_impl {
    struct gm_logger* log;

    JSON_Value* data_meta;
    JSON_Value* record;
    JSON_Value* history;

    char*    reload;        // Reload and continue training with pre-existing tree

    pthread_mutex_t tidy_log_lock;
    bool     verbose;       // Verbose logging
    bool     profile;       // Verbose profiling
    bool     debug_post_inference; // After training, write label inference images
                                   // for sampled pixels across all images

    bool     pretty;        // Pretty JSON output
    int      seed;          // Seed for RNG

    char*    data_dir;      // Location of training data
    char*    index_name;    // Name of the frame index like <name>.index to read
    char*    out_filename;  // Filename of tree (.json or .rdt) to write

    float    fov;           // Camera field of view
    int      n_labels;      // Number of labels in label images

    int      n_images;      // Number of training images

    uint64_t  last_load_update;
    int16_t* depth_images;  // Depth images (row-major)

    int      n_uvs;         // Number of combinations of u,v pairs
    float    uv_range;      // Range of u,v combinations to generate
    int      n_thresholds;  // The number of thresholds
    float    threshold_range;       // Range of thresholds to test (in meters)
    int      max_depth;     // Maximum depth to train to
    int      max_nodes;     // Maximum number of nodes to train - used for debug
                            // and testing to trigger an early exit.
    int        n_pixels;      // Number of pixels to sample
    std::vector<int32_t> uvs; // The uv pairs to test ordered like:
                              // [uv0.x, uv0.y, uv1.x, uv1.y]
                              // values are in pixel-millimeter units
    int16_t* thresholds;    // A list of thresholds to test

    int      n_threads;     // How many threads to spawn for training

    int      n_nodes_trained;   // The number of nodes trained so far

    int      uvt_histograms_mem; // Constraint on working set memory usage for
                                 // UVT left/right histograms

    uint64_t start;

    std::vector<depth_meta> depth_index;

    std::vector<thread_state> thread_pool;

    pthread_mutex_t     scheduler_lock;

    pthread_mutex_t         train_queue_lock;
    std::deque<node_data>   train_queue; // deque so we can iterate for debugging

    // Queue of work for thread pool
    pthread_mutex_t     work_queue_lock;
    pthread_cond_t      work_queue_changed;
    std::deque<work>    work_queue; //deque so we can iterate for debugging
    int                 n_idle; // number of threads currently waiting for work

    std::vector<node>   tree; // The decision tree being built
    std::vector<float>  tree_histograms; // label histograms for leaf nodes

    std::vector<uint32_t>  root_pixel_histogram; // label histogram for initial pixels
    std::vector<float>  root_pixel_nhistogram; // normalized histogram for initial pixels

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
};

static void
process_node_shards_work_cb(struct thread_state* state,
                            void* user_data);


static const char *interrupt_reason;
static bool interrupted;

static png_color png_label_palette[] = {
    { 0x21, 0x21, 0x21 },
    { 0xd1, 0x15, 0x40 },
    { 0xda, 0x1d, 0x0e },
    { 0xdd, 0x5d, 0x1e },
    { 0x49, 0xa2, 0x24 },
    { 0x29, 0xdc, 0xe3 },
    { 0x02, 0x68, 0xc2 },
    { 0x90, 0x29, 0xf9 },
    { 0xff, 0x00, 0xcf },
    { 0xef, 0xd2, 0x37 },
    { 0x92, 0xa1, 0x3a },
    { 0x48, 0x21, 0xeb },
    { 0x2f, 0x93, 0xe5 },
    { 0x1d, 0x6b, 0x0e },
    { 0x07, 0x66, 0x4b },
    { 0xfc, 0xaa, 0x98 },
    { 0xb6, 0x85, 0x91 },
    { 0xab, 0xae, 0xf1 },
    { 0x5c, 0x62, 0xe0 },
    { 0x48, 0xf7, 0x36 },
    { 0xa3, 0x63, 0x0d },
    { 0x78, 0x1d, 0x07 },
    { 0x5e, 0x3c, 0x00 },
    { 0x9f, 0x9f, 0x60 },
    { 0x51, 0x76, 0x44 },
    { 0xd4, 0x6d, 0x46 },
    { 0xff, 0xfb, 0x7e },
    { 0xd8, 0x4b, 0x4b },
    { 0xa9, 0x02, 0x52 },
    { 0x0f, 0xc1, 0x66 },
    { 0x2b, 0x5e, 0x44 },
    { 0x00, 0x9c, 0xad },
    { 0x00, 0x40, 0xad },
    { 0xff, 0x5d, 0xaa },
};



static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* The longest format is like "00:00:00" which needs up to 9 bytes but notably
 * gcc complains if buf < 14 bytes, so rounding up to power of two for neatness.
 */
char *
format_duration_s16(uint64_t duration_ns, char buf[16])
{
    if (duration_ns > 1000000000) {
        const uint64_t hour_ns = 1000000000ULL*60*60;
        const uint64_t min_ns = 1000000000ULL*60;
        const uint64_t sec_ns = 1000000000ULL;

        uint64_t hours = duration_ns / hour_ns;
        duration_ns -= hours * hour_ns;
        uint64_t minutes = duration_ns / min_ns;
        duration_ns -= minutes * min_ns;
        uint64_t seconds = duration_ns / sec_ns;
        snprintf(buf, 16, "%02d:%02d:%02d", (int)hours, (int)minutes, (int)seconds);
    } else if (duration_ns > 1000000) {
        uint64_t ms = duration_ns / 1000000;
        snprintf(buf, 16, "%dms", (int)ms);
    } else if (duration_ns > 1000) {
        uint64_t us = duration_ns / 1000;
        snprintf(buf, 16, "%dus", (int)us);
    } else {
        snprintf(buf, 16, "%dns", (int)duration_ns);
    }

    return buf;
}

static void
print_label_histogram(struct gm_logger* log,
                      JSON_Array* labels,
                      float* histogram)
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

static void
calculate_thread_depth_metrics_report(struct thread_state* state,
                                      int depth,
                                      uint64_t partial_work_time,
                                      struct thread_depth_metrics_raw* raw,
                                      struct thread_depth_metrics_report* metrics)
{
    uint64_t run_duration = raw->idle_time + raw->work_time + partial_work_time;

    memset(metrics, 0, sizeof(*metrics));
    if (!run_duration)
        return;

    double idle_percent = ((double)raw->idle_time / run_duration) * 100.0;
    double accu_percent = ((double)raw->accumulation_time / run_duration) * 100.0;
    double rank_percent = ((double)raw->gain_ranking_time / run_duration) * 100.0;

    double run_time_sec = run_duration / 1e9;
    double nodes_per_second = (double)raw->n_nodes / run_time_sec;
    double images_per_sec = (double)raw->n_images_accumulated / run_time_sec;
    double px_per_sec = (double)raw->n_pixels_accumulated / run_time_sec;
    double uvs_per_sec = (double)raw->n_uvs_accumulated / run_time_sec;
    double thresholds_per_sec = (double)raw->n_thresholds_accumulated / run_time_sec;

    metrics->duration = run_duration;

    metrics->idle_percent = idle_percent;
    metrics->accumulation_percent = accu_percent;
    metrics->ranking_percent = rank_percent;

    metrics->nodes_per_second = nodes_per_second;
    metrics->images_per_second = images_per_sec;
    metrics->pixels_per_second = px_per_sec;
    metrics->uvs_per_second = uvs_per_sec;
    metrics->thresholds_per_second = thresholds_per_sec;
}

static JSON_Value*
thread_metrics_to_json(struct gm_rdt_context_impl* ctx,
                       struct thread_depth_metrics_report* metrics)
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
log_thread_depth_metrics(struct gm_rdt_context_impl *ctx,
                         const char *prefix,
                         int depth,
                         struct thread_depth_metrics_report *metrics)
{
    char buf[16];

    gm_info(ctx->log, "%s%-2d: taken %8s: idle %5.2f%%, acc %5.2f%% (%5.2f nd/s %6d img/s, %7d px/s, %7d uvs/s, %7d thresh/s), ranking %5.2f%%",
            prefix,
            depth,

            format_duration_s16(metrics->duration, buf),

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
maybe_log_thread_depth_metrics(struct thread_state* state,
                               int n_node_pixels,
                               int depth,
                               int n_shards,
                               uint64_t current_time)
{
    if (current_time - state->last_metrics_log > 5000000000) {
        struct gm_rdt_context_impl* ctx = state->ctx;
        uint64_t partial_work_time = current_time - state->current_work_start;

        struct thread_depth_metrics_raw* raw =
            &state->per_depth_metrics[depth];

        // Assume an even distribution of work across threads and shards
        // to estimate the current node progress...
        int nodes_per_depth = 1<<depth;
        int64_t progress = ((raw->n_pixels_accumulated/nodes_per_depth) * 100 /
                            ((int64_t)n_node_pixels * n_shards / ctx->n_threads));

        char prefix[256];
        snprintf(prefix, sizeof(prefix), "Thread %2d, node %2d%% (%" PRIu64 "/%" PRIu64 ", %d shards), d",
                 state->idx,
                 (int)progress,
                 raw->n_pixels_accumulated / nodes_per_depth,
                 (int64_t)n_node_pixels * n_shards / ctx->n_threads,
                 n_shards);

        struct thread_depth_metrics_report metrics;

        calculate_thread_depth_metrics_report(state, depth,
                                              partial_work_time,
                                              raw,
                                              &metrics);
        pthread_mutex_lock(&ctx->tidy_log_lock);
        log_thread_depth_metrics(ctx, prefix, depth, &metrics);
        pthread_mutex_unlock(&ctx->tidy_log_lock);
        state->last_metrics_log = current_time;
    }
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
     * Our node IDs are base 0 (i.e. root node has ID = 0) but with base-1
     * IDs then every depth level neatly starts with a power of two ID (thus
     * the (id32 + 1))
     *
     * Finally subtract 1 to return a base-0 ID.
     */
    return 32 - __builtin_clz(id32 + 1) - 1;
}

struct labels_pre_processor
{
    struct gm_rdt_context_impl* ctx;
    uint64_t last_update;
    std::mt19937 rng;
    std::uniform_real_distribution<float> rand_0_1;
    int width;
    int height;
    int n_image_pixels;
    std::vector<uint8_t> image_buf;
    std::vector<int> in_body_pixels;
    std::vector<int> indices;
    struct bounds* body_bounds;
    struct pixel* random_pixels;
};

static bool
pre_process_label_image_cb(struct gm_data_index* data_index,
                           int index,
                           const char* frame_path,
                           void* user_data,
                           char** err)
{
    struct labels_pre_processor* labels_pre_processor =
        (struct labels_pre_processor*)user_data;
    struct gm_rdt_context_impl* ctx = labels_pre_processor->ctx;
    uint8_t* label_image = labels_pre_processor->image_buf.data();
    int width = labels_pre_processor->width;
    int height = labels_pre_processor->height;

    const char* top_dir = gm_data_index_get_top_dir(data_index);

    char labels_filename[512];
    xsnprintf(labels_filename, sizeof(labels_filename), "%s/labels/%s.png",
              top_dir, frame_path);

    IUImageSpec label_spec = { width, height, IU_FORMAT_U8 };
    if (iu_read_png_from_file(labels_filename, &label_spec, &label_image,
                              NULL, // palette output
                              NULL) // palette size
        != SUCCESS)
    {
        gm_throw(ctx->log, err, "Failed to read image '%s'\n", labels_filename);
        return false;
    }

    /* Our tracking system assumes that the body has been segmented
     * from the background before we try and label the different parts
     * of the body and so we're only interested in sampling points
     * inside the body...
     */
    labels_pre_processor->in_body_pixels.clear();

    struct bounds bounds;
    bounds.min_x = INT_MAX;
    bounds.max_x = 0;
    bounds.min_y = INT_MAX;
    bounds.max_y = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int off = y * width + x;
            int label = (int)label_image[off];

            gm_assert(ctx->log, label < ctx->n_labels,
                      "Label '%d' is bigger than expected (max %d)\n",
                      label, ctx->n_labels - 1);

            if (label != 0) { // 0 = background
                if (x < bounds.min_x)
                    bounds.min_x = x;
                if (x > bounds.max_x)
                    bounds.max_x = x;
                if (y < bounds.min_y)
                    bounds.min_y = y;
                if (y > bounds.max_y)
                    bounds.max_y = y;

                labels_pre_processor->in_body_pixels.push_back(off);
            }
        }
    }

    /* The image-pre-processor tool should already check this so just have
     * a *very* conservative sanity check here...
     */
    gm_assert(ctx->log, labels_pre_processor->in_body_pixels.size() > 100,
              "Fewer than 100 non-background pixels found in frame %s",
              labels_filename);

    int crop_width = bounds.max_x - bounds.min_x + 1;
    int crop_height = bounds.max_y - bounds.min_y + 1;

    gm_assert(ctx->log, width <= width,
              "Bounded width (%d) > full width (%d)", crop_width, width);
    gm_assert(ctx->log, height <= height,
              "Bounded height (%d) > full height (%d)", crop_height, height);

    labels_pre_processor->body_bounds[index] = bounds;

    /* Note: we don't do anything to filter out duplicates which could
     * be fairly likely for frames where the body is relatively small.
     *
     * It seems best to not bias how many samples we consider across
     * the body based on the in-frame size, so our training expends
     * approximately the same amount of energy training on each pose
     * regardless of body size or distance from the camera.
     */
    int n_body_points = labels_pre_processor->in_body_pixels.size();
    labels_pre_processor->indices.clear();
    for (int j = 0; j < ctx->n_pixels; j++) {

        int off = labels_pre_processor->rand_0_1(labels_pre_processor->rng) * n_body_points;

        /* XXX: It's important we clamp here since the rounding can otherwise
         * result in off == n_body_points */
        labels_pre_processor->indices.push_back(std::min(off, n_body_points - 1));
    }

    /* May slightly improve cache access patterns if we can process
     * our samples in memory order, even though the UV sampling
     * is somewhat randomized relative to these pixels...
     */
    std::sort(labels_pre_processor->indices.begin(),
              labels_pre_processor->indices.end());

    for (int j = 0; j < ctx->n_pixels; j++) {
        int off = labels_pre_processor->in_body_pixels[labels_pre_processor->indices[j]];

        struct pixel pixel;
        pixel.x = off % width;
        pixel.y = off / width;
        pixel.label = label_image[off];
        pixel.i = index;

        /* Try and double check we haven't muddled up something silly... */
        gm_assert(ctx->log, label_image[pixel.y * width + pixel.x] != 0,
                  "Spurious background pixel (%d,%d) sampled for image %s",
                  pixel.x, pixel.y,
                  frame_path);

        pixel.x -= bounds.min_x;
        pixel.y -= bounds.min_y;

        gm_assert(ctx->log, pixel.x < crop_width,
                  "Image %s (%d,%d) has out-of-bounds width %d (after cropping to +%d,+%d,%dx%d)",
                  frame_path,
                  pixel.x + bounds.min_x,
                  pixel.y + bounds.min_y,
                  pixel.x,
                  bounds.min_x, bounds.min_y,
                  crop_width, crop_height);
        gm_assert(ctx->log, pixel.y < crop_height,
                  "Image %s (%d,%d) has out-of-bounds height %d (after cropping to +%d,+%d,%dx%d)",
                  frame_path,
                  pixel.x + bounds.min_x,
                  pixel.y + bounds.min_y,
                  pixel.y,
                  bounds.min_x, bounds.min_y,
                  crop_width, crop_height);

        labels_pre_processor->random_pixels[(int64_t)index * ctx->n_pixels + j] = pixel;
    }

    uint64_t current = get_time();
    if (current - labels_pre_processor->last_update > 2000000000) {
        int percent = index * 100 / ctx->n_images;
        gm_info(ctx->log, "%3d%%", percent);
        labels_pre_processor->last_update = current;
    }

    return true;
}

/* For every image, pick N (ctx->n_pixels) random points within the silhoette
 * of the example pose for that frame.
 */
static bool
pre_process_label_images(struct gm_rdt_context_impl* ctx,
                         struct gm_data_index* data_index,
                         struct pixel* random_pixels, /* len: n_images * n_pixels */
                         struct bounds* body_bounds, /* len: n_images */
                         char** err)
{
    struct labels_pre_processor labels_pre_processor;

    labels_pre_processor.ctx = ctx;
    labels_pre_processor.last_update = get_time();
    labels_pre_processor.rng = std::mt19937(ctx->seed);
    labels_pre_processor.rand_0_1 = std::uniform_real_distribution<float>(0.0, 1.0);
    labels_pre_processor.width = gm_data_index_get_width(data_index);
    labels_pre_processor.height = gm_data_index_get_height(data_index);
    int n_image_pixels = labels_pre_processor.width * labels_pre_processor.height;
    labels_pre_processor.image_buf = std::vector<uint8_t>(n_image_pixels);
    labels_pre_processor.in_body_pixels = std::vector<int>(n_image_pixels);
    labels_pre_processor.indices = std::vector<int>(n_image_pixels);
    labels_pre_processor.random_pixels = random_pixels;
    labels_pre_processor.body_bounds = body_bounds;

    gm_info(ctx->log, "Randomly sampling training pixels (%d per-image) across %d images...",
            ctx->n_pixels, ctx->n_images);
    if (!gm_data_index_foreach(data_index,
                               pre_process_label_image_cb,
                               &labels_pre_processor,
                               err))
    {
        return false;
    }

    return true;
}

static void
normalize_histogram_16(uint16_t* histogram,
                       int n_labels,
                       float* normalized,
                       int* n_histogram_pixels_ret,
                       int* n_histogram_labels_ret)
{
    int n_histogram_pixels = 0;
    int n_histogram_labels = 0;

    for (int i = 0; i < n_labels; i++) {
        if (histogram[i] > 0) {
            n_histogram_pixels += histogram[i];
            n_histogram_labels++;
        }
    }

    if (n_histogram_pixels) {
        for (int i = 0; i < n_labels; i++)
            normalized[i] = histogram[i] / (float)n_histogram_pixels;
    } else
        memset(normalized, 0, n_labels * sizeof(float));

    *n_histogram_pixels_ret = n_histogram_pixels;
    *n_histogram_labels_ret = n_histogram_labels;
}

static void
normalize_histogram_32(uint32_t* histogram,
                       int n_labels,
                       float* normalized,
                       int* n_histogram_pixels_ret,
                       int* n_histogram_labels_ret)
{
    int n_histogram_pixels = 0;
    int n_histogram_labels = 0;

    for (int i = 0; i < n_labels; i++) {
        if (histogram[i] > 0) {
            n_histogram_pixels += histogram[i];
            n_histogram_labels++;
        }
    }

    if (n_histogram_pixels) {
        for (int i = 0; i < n_labels; i++)
            normalized[i] = histogram[i] / (float)n_histogram_pixels;
    } else
        memset(normalized, 0, n_labels * sizeof(float));

    *n_histogram_pixels_ret = n_histogram_pixels;
    *n_histogram_labels_ret = n_histogram_labels;
}

static inline float
calculate_shannon_entropy(float* normalized_histogram, int n_labels)
{
    float entropy = 0.f;
    for (int i = 0; i < n_labels; i++) {
        float value = normalized_histogram[i];
        if (value > 0.f && value < 1.f)
            entropy += -value * log2f(value);
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
accumulate_pixels_histogram_32(struct gm_rdt_context_impl* ctx,
                               struct node_data* data,
                               uint32_t* node_histogram)
{
    int n_pixels = data->n_pixels;

    for (int p = 0; p < n_pixels; p++) {
        int label = data->pixels[p].label;

        node_histogram[label]++;

        if (p % 10000 == 0) {
            if (interrupted)
                break;
        }
    }
}

/* Implement a fixed point round-nearest, considering that the numerator
 * might be negative.
 *
 * XXX: this assumes the denominator is never negative!
 * XXX: this assumes n is not close to the maximum value of the int type
 */
#define div_int_round_nearest(N, D, HALF_D) \
    ((N < 0) ? ((N - HALF_D)/D) : ((N + HALF_D)/D))

static inline int16_t
sample_uv_gradient_mm(int16_t* depth_image,
                      int16_t width,
                      int16_t height,
                      int16_t x,
                      int16_t y,
                      int16_t depth_mm,
                      int16_t half_depth_mm,
                      int32_t* uvs)
{
    int32_t u_x = x + div_int_round_nearest(uvs[0], depth_mm, half_depth_mm);
    int32_t u_y = y + div_int_round_nearest(uvs[1], depth_mm, half_depth_mm);
    int32_t v_x = x + div_int_round_nearest(uvs[2], depth_mm, half_depth_mm);
    int32_t v_y = y + div_int_round_nearest(uvs[3], depth_mm, half_depth_mm);

    int16_t u_z = (u_x >= 0 && u_x < width &&
                   u_y >= 0 && u_y < height) ?
        depth_image[(u_y * width + u_x)] : INT16_MAX;
    int16_t v_z = (v_x >= 0 && v_x < width &&
                   v_y >= 0 && v_y < height) ?
        depth_image[(v_y * width + v_x)] : INT16_MAX;
    //gm_info(log, "u=(%d,%d,%d), v=(%d,%d,%d), grad=%d",
    //        u_x - x, u_y - y, u_z, v_x - x, v_y - y, v_z, u_z - v_z);

    return u_z - v_z;
}

static void
accumulate_uvt_lr_histograms(struct gm_rdt_context_impl* ctx,
                             struct thread_state *state,
                             struct node_data* data,
                             int uv_start, int uv_end,
                             int n_shards)
{
    int p;
    int last_i = -1;
    int max_depth = ctx->max_depth;
    int node_depth = id_to_depth(data->id);
    struct depth_meta* depth_index = ctx->depth_index.data();
    int n_pixels = data->n_pixels;
    int n_labels = ctx->n_labels;
    int n_thresholds = ctx->n_thresholds;

    struct thread_depth_metrics_raw *depth_metrics =
        &state->per_depth_metrics[node_depth];

    uint16_t* uvt_lr_histograms_16 = state->uvt_lr_histograms_16.data();
    uint32_t* uvt_lr_histograms_32 = state->uvt_lr_histograms_32.data();

    struct depth_meta depth_meta = {};

    for (p = 0; p < n_pixels; p++)
    {
        struct pixel px = data->pixels[p];
        int i = px.i;
        int label = px.label;

        depth_metrics->n_pixels_accumulated++;
        if (i != last_i) {
            depth_meta = depth_index[i];
            depth_metrics->n_images_accumulated++;
            last_i = i;
        }

        int n_uv_combos = uv_end - uv_start;

        if (p % 10000 == 0) {
            if (interrupted)
                break;
            if (ctx->profile) {
                maybe_log_thread_depth_metrics(state,
                                               n_pixels,
                                               node_depth,
                                               n_shards,
                                               get_time());
            }
        }

        gm_assert(ctx->log, label < n_labels,
                  "Label '%d' is bigger than expected (max %d)\n",
                  label, n_labels - 1);

        // Don't waste processing time if this is the last depth
        if (node_depth >= max_depth - 1) {
            continue;
        }

        // Accumulate LR branch histograms

        int16_t* depth_image = &ctx->depth_images[depth_meta.pixel_offset];

        int16_t depth_mm = depth_image[px.y * depth_meta.width + px.x];
        int16_t half_depth = depth_mm / 2;

        int32_t *uvs = ctx->uvs.data();

        int16_t gradients[n_uv_combos];
        for (int c = uv_start; c < uv_end; c++) {
            gradients[c - uv_start] = sample_uv_gradient_mm(depth_image,
                                                            depth_meta.width,
                                                            depth_meta.height,
                                                            px.x, px.y,
                                                            depth_mm,
                                                            half_depth,
                                                            uvs + 4 * c);
        }

        /* Aim to minimize our memory bandwidth usage here by using 16bit
         * histograms, since this is our typical bottleneck...
         */
        if (n_pixels < UINT16_MAX) {
            for (int i = 0;  i < n_uv_combos; i++) {
                int uv_offset = i * n_thresholds * n_labels * 2;
                for (int n = 0; n < n_thresholds; n++) {
                    int threshold = ctx->thresholds[n];
                    int t_offset = n * n_labels * 2;
                    int lr_histogram_idx = uv_offset + t_offset;

                    //gm_info(ctx->log, "combo %d: %s", c,
                    //        gradients[c] < threshold ? "left" : "right");
                    // Accumulate histogram for this particular uvt combination
                    // on both theoretical branches
                    ++uvt_lr_histograms_16[(gradients[i] < threshold) ?
                        lr_histogram_idx + label :
                        lr_histogram_idx + n_labels + label];
                }
            }
        } else {
            for (int i = 0;  i < n_uv_combos; i++) {
                int uv_offset = i * n_thresholds * n_labels * 2;
                for (int n = 0; n < n_thresholds; n++) {
                    int threshold = ctx->thresholds[n];
                    int t_offset = n * n_labels * 2;
                    int lr_histogram_idx = uv_offset + t_offset;

                    //gm_info(ctx->log, "combo %d: %s", c,
                    //        gradients[c] < threshold ? "left" : "right");
                    // Accumulate histogram for this particular uvt combination
                    // on both theoretical branches
                    ++uvt_lr_histograms_32[(gradients[i] < threshold) ?
                        lr_histogram_idx + label :
                        lr_histogram_idx + n_labels + label];
                }
            }
        }
    }
}

static void
node_shard_work_cb(struct thread_state* state,
                   void* user_data)
{
    struct gm_rdt_context_impl* ctx = state->ctx;
    struct node_shard_work* shard_work = (struct node_shard_work*)user_data;

    struct node_shard_results* results = shard_work->results;
    struct node_shard_data* shard_data = &results->data[shard_work->shard_index];

    int n_labels = ctx->n_labels;

    struct node_data node_data = shard_work->node_data;

    // Histograms for each uvt combination being tested
    int n_uv_combos = shard_work->uv_end - shard_work->uv_start;
    int n_thresholds = ctx->n_thresholds;
    int n_uvt_combos = n_uv_combos * n_thresholds;
    if (node_data.n_pixels < UINT16_MAX) {
        state->uvt_lr_histograms_16.clear();
        state->uvt_lr_histograms_16.resize(ctx->n_labels * n_uvt_combos * 2);
    } else {
        state->uvt_lr_histograms_32.clear();
        state->uvt_lr_histograms_32.resize(ctx->n_labels * n_uvt_combos * 2);
    }

    uint16_t* uvt_lr_histograms_16 = state->uvt_lr_histograms_16.data();
    uint32_t* uvt_lr_histograms_32 = state->uvt_lr_histograms_32.data();

    int node_depth = id_to_depth(node_data.id);
    struct thread_depth_metrics_raw *depth_metrics =
        &state->per_depth_metrics[node_depth];

    if (ctx->verbose) {
        gm_info(ctx->log, "Training shard %d of node %d, depth=%d",
                shard_work->shard_index, node_data.id, node_depth);
    }

    // Determine the best u,v,t combination
    shard_data->best_gain = 0.f;

    // If there's only 1 label, skip all this, gain is zero
    if (results->n_node_labels > 1 && node_depth < ctx->max_depth - 1)
    {
        uint64_t accu_start = get_time();
        accumulate_uvt_lr_histograms(ctx,
                                     state,
                                     &node_data,
                                     shard_work->uv_start, shard_work->uv_end,
                                     results->n_shards);
        uint64_t accu_end = get_time();
        depth_metrics->accumulation_time += accu_end - accu_start;


        uint64_t rank_start = get_time();

        // Calculate the shannon entropy for the normalised label histogram
        float entropy = calculate_shannon_entropy(results->nhistogram,
                                                  ctx->n_labels);

        int n_uv_combos = shard_work->uv_end - shard_work->uv_start;
        // Calculate the gain for each combination of u,v,t and store the best
        for (int i = 0; i < n_uv_combos; i++) {
            int uv_offset = i * n_thresholds * n_labels * 2;

            for (int j = 0; j < ctx->n_thresholds && !interrupted; j++) {
                int t_offset = j * n_labels * 2;
                int lr_histo_base = uv_offset + t_offset;
                float nhistogram[ctx->n_labels];
                float l_entropy, r_entropy, gain;

                int l_n_pixels = 0;
                int l_n_labels = 0;

                if (node_data.n_pixels < UINT16_MAX) {
                    normalize_histogram_16(&uvt_lr_histograms_16[lr_histo_base],
                                           ctx->n_labels, nhistogram,
                                           &l_n_pixels,
                                           &l_n_labels);
                } else {
                    normalize_histogram_32(&uvt_lr_histograms_32[lr_histo_base],
                                           ctx->n_labels, nhistogram,
                                           &l_n_pixels,
                                           &l_n_labels);
                }
                if (l_n_pixels == 0 || l_n_pixels == node_data.n_pixels)
                    continue;

                l_entropy = calculate_shannon_entropy(nhistogram,
                                                      ctx->n_labels);

                int r_n_pixels = 0;
                int r_n_labels = 0;
                if (node_data.n_pixels < UINT16_MAX) {
                    normalize_histogram_16(
                        &uvt_lr_histograms_16[lr_histo_base + ctx->n_labels],
                        ctx->n_labels, nhistogram,
                        &r_n_pixels,
                        &r_n_labels);
                } else {
                    normalize_histogram_32(
                        &uvt_lr_histograms_32[lr_histo_base + ctx->n_labels],
                        ctx->n_labels, nhistogram,
                        &r_n_pixels,
                        &r_n_labels);
                }
                r_entropy = calculate_shannon_entropy(nhistogram,
                                                      ctx->n_labels);

                gain = calculate_gain(entropy, node_data.n_pixels,
                                      l_entropy, l_n_pixels,
                                      r_entropy, r_n_pixels);

                if (gain > shard_data->best_gain) {
                    shard_data->best_gain = gain;
                    shard_data->best_uv = shard_work->uv_start + i;
                    shard_data->best_threshold = j;
                    shard_data->n_lr_pixels[0] = l_n_pixels;
                    shard_data->n_lr_pixels[1] = r_n_pixels;
                }
            }
        }
        uint64_t rank_end = get_time();
        depth_metrics->gain_ranking_time += rank_end - rank_start;
    }

    depth_metrics->n_uvs_accumulated += n_uv_combos;
    depth_metrics->n_thresholds_accumulated += n_uvt_combos;
    shard_data->done = true;

    /* Once we set shard_data->done then we are racing with other threads
     * that might process the results if this was the last node.
     *
     * NB: The shard results are ref counted and we handle unrefs in
     * process_node_shards_work_cb
     */

    if (ctx->verbose) {
        gm_info(ctx->log, "Queueing done follow up for %d", node_data.id);
    }
    struct process_node_shards_work* process_work =
        (struct process_node_shards_work*)xmalloc(sizeof(*process_work));
    process_work->node_data = node_data;
    process_work->results = shard_work->results;

    struct work entry;
    entry.depth = node_depth;
    entry.work_cb = process_node_shards_work_cb;
    entry.user_data = process_work;

    pthread_mutex_lock(&ctx->work_queue_lock);
    ctx->work_queue.push_back(entry);
    pthread_cond_broadcast(&ctx->work_queue_changed);
    pthread_mutex_unlock(&ctx->work_queue_lock);

    xfree(shard_work);
    // Note: the shard_work->results will be unreferenced/freed by
    // process_node_shards_work_cb
}

static bool
schedule_node_work(struct thread_state* state)
{
    struct gm_rdt_context_impl* ctx = state->ctx;
    struct node_data node_data;

    pthread_mutex_lock(&ctx->scheduler_lock);

    bool busy = false;
    bool popped_node = false;

    pthread_mutex_lock(&ctx->work_queue_lock);
    if ((int)ctx->work_queue.size() >= (ctx->n_threads * 2)) {
        if (ctx->verbose) {
            gm_info(ctx->log, "Work queue len %d > %d, therefore too busy to schedule more work",
                    (int)ctx->work_queue.size(), (int)(ctx->n_threads * 2));
        }
        busy = true;
    }
    pthread_mutex_unlock(&ctx->work_queue_lock);

    if (!busy) {
        pthread_mutex_lock(&ctx->train_queue_lock);
        if (ctx->verbose) {
            gm_info(ctx->log, "Training queue len = %d:",
                    (int)ctx->train_queue.size());
            int first_n = std::min(25, (int)ctx->train_queue.size());
            for (int i = 0; i < first_n; i++) {
                struct node_data node_tmp = ctx->train_queue[i];
                gm_info(ctx->log, "  id=%d, n_pixels=%d",
                        node_tmp.id, node_tmp.n_pixels);
            }
            if (first_n == 25)
                gm_info(ctx->log, "  ...");
        }
        if (!ctx->train_queue.empty()) {
            node_data = ctx->train_queue.front();
            ctx->train_queue.pop_front();
            popped_node = true;
        }
        pthread_mutex_unlock(&ctx->train_queue_lock);
    }

    pthread_mutex_unlock(&ctx->scheduler_lock);

    /* We don't block waiting for something to schedule because we can assume
     * some other thread will schedule work after adding nodes to the training
     * queue (or else we will recognise we have finished when all threads are
     * idle)
     */
    if (!popped_node) {
        if (ctx->verbose) {
            if (busy) {
                gm_info(ctx->log, "Currently too busy to schedule");
            } else {
                gm_info(ctx->log, "Currently no work to schedule");
            }
        }
        return false;
    }

    /*
     * XXX: Note that because we don't synchronize the full scheduler then
     * multiple threads may reach this point concurrently and submit more
     * work in total than the above 'busy' threshold. This isn't expected
     * to be a problem though since there's still a reasonable bound
     * (n_threads * max_shards) on how much can be scheduled and work entries
     * aren't very large.
     */

    int node_depth = id_to_depth(node_data.id);

    /*
     * TODO: It's probably worth implementing some simple caching allocator for
     * node_shard_results and node_shard_work structures, considering the
     * maximum degree of sharding and the maximum number of in-flight nodes.
     */

    /* We must calculate results->n_shards before scheduling any of the
     * individual shards, otherwise there will be a race and a subset of the
     * shards may be considered complete (and the results processed) before we
     * finish submitting all of the shards to the work queue.
     *
     * We also want to schedule all the shards atomically so verbose printing
     * of the work queue changes won't race with other worker threads taking
     * jobs from the queue.
     */

    // We want the working set of uvt combos to be constrained enough that
    // the uvt_lr_histrograms array can be cached
    int est_uvt_lr_hist_size = ctx->n_uvs * ctx->n_thresholds * ctx->n_labels * 4 * 2;
    int max_thread_uvt_lr_size =
        (std::min(ctx->uvt_histograms_mem, est_uvt_lr_hist_size) /
         ctx->n_threads);
    int n_shards = est_uvt_lr_hist_size / max_thread_uvt_lr_size;
    int n_uvs_per_shard = std::max(ctx->n_uvs / n_shards, 1);
    n_shards = ctx->n_uvs / n_uvs_per_shard;

    size_t node_data_size = sizeof(struct node_shard_data) * n_shards;
    struct node_shard_results* node_results =
        (struct node_shard_results*)xcalloc(1, sizeof(*node_results) +
                                            node_data_size);
    pthread_mutex_init(&node_results->check_lock, NULL);

    node_results->n_shards = n_shards;
    node_results->ref = n_shards;

    // Histogram for the node being processed
    uint32_t node_histogram[ctx->n_labels];
    memset(node_histogram, 0, sizeof(node_histogram));

    accumulate_pixels_histogram_32(ctx, &node_data, node_histogram);

    // Calculate the normalised label histogram and get the number of pixels
    // and the number of labels in the root histogram.
    int n_node_pixels = 0;
    normalize_histogram_32(node_histogram,
                           ctx->n_labels,
                           node_results->nhistogram,
                           &n_node_pixels,
                           &node_results->n_node_labels);

    gm_assert(ctx->log, n_node_pixels == node_data.n_pixels,
              "Mismatching N pixels from node_data (%d) and histogram (%d)",
              node_data.n_pixels, n_node_pixels);

    if (ctx->verbose) {
        gm_info(ctx->log, "Scheduling node %d with %d pixels, histogram:",
                node_data.id,
                n_node_pixels);
        JSON_Array* labels =
            json_object_get_array(json_object(ctx->data_meta), "labels");
        print_label_histogram(ctx->log, labels, node_results->nhistogram);
    }

    struct work jobs[n_shards];

    for (int i = 0; i < n_shards; i++) {
        struct node_shard_work *node_work =
            (struct node_shard_work*)xmalloc(sizeof(*node_work));

        node_work->node_data = node_data;
        node_work->uv_start = i * n_uvs_per_shard;
        int end = (i + 1) * n_uvs_per_shard;
        if (i == (n_shards - 1) || end > ctx->n_uvs)
            end = ctx->n_uvs;
        node_work->uv_end = end;
        node_work->results = node_results;
        node_work->shard_index = i;

        jobs[i].depth = node_depth;
        jobs[i].work_cb = node_shard_work_cb;
        jobs[i].user_data = node_work;
    }

    pthread_mutex_lock(&ctx->work_queue_lock);
    for (int i = 0; i < n_shards; i++)
        ctx->work_queue.push_back(jobs[i]);

    if (ctx->verbose) {
        pthread_mutex_lock(&ctx->tidy_log_lock);
        gm_info(ctx->log, "work queue:");
        int i = 0;
        for (auto &iter: ctx->work_queue) {
            if (iter.work_cb == node_shard_work_cb) {
                struct node_shard_work *work = (struct node_shard_work*)iter.user_data;
                gm_info(ctx->log, "  %-3d: shard,           node id=%d, depth=%d, shard=%d",
                        i, work->node_data.id, iter.depth, work->shard_index);
            } else if (iter.work_cb == process_node_shards_work_cb) {
                struct process_node_shards_work *work =
                    (struct process_node_shards_work*)iter.user_data;
                gm_info(ctx->log, "  %-3d: process results, node id=%d, depth=%d",
                        i, work->node_data.id, iter.depth);
            } else
                gm_info(ctx->log, "  %-3d: unknown", i);
            i++;
        }
        pthread_mutex_unlock(&ctx->tidy_log_lock);
    }

    pthread_cond_broadcast(&ctx->work_queue_changed);
    pthread_mutex_unlock(&ctx->work_queue_lock);

    return true;
}

static void
collect_pixels(struct gm_rdt_context_impl* ctx,
               struct node_data* data,
               int32_t* uvs,
               int16_t t_mm,
               struct pixel** l_pixels,
               struct pixel** r_pixels,
               int* n_lr_pixels)
{
    *l_pixels = (struct pixel*)xmalloc((n_lr_pixels[0] ? n_lr_pixels[0] :
                                        data->n_pixels) *
                                       sizeof(struct pixel));
    *r_pixels = (struct pixel*)xmalloc((n_lr_pixels[1] ? n_lr_pixels[1] :
                                        data->n_pixels) *
                                       sizeof(struct pixel));

    int l_index = 0;
    int r_index = 0;

    struct depth_meta* depth_index = ctx->depth_index.data();
    int16_t* depth_images = ctx->depth_images;
    for (int p = 0; p < data->n_pixels; p++) {
        struct pixel px = data->pixels[p];

        struct depth_meta depth_meta = depth_index[px.i];
        int16_t* depth_image = &depth_images[depth_meta.pixel_offset];

        int16_t depth_mm = depth_image[px.y * depth_meta.width + px.x];
        int16_t gradient = sample_uv_gradient_mm(depth_image,
                                                 depth_meta.width,
                                                 depth_meta.height,
                                                 px.x, px.y,
                                                 depth_mm,
                                                 depth_mm / 2,
                                                 uvs);
        if (gradient < t_mm)
            (*l_pixels)[l_index++] = px;
        else
            (*r_pixels)[r_index++] = px;
    }

    if (n_lr_pixels[0] != l_index) {
        *l_pixels = (struct pixel*)xrealloc(*l_pixels,
                                            l_index * sizeof(struct pixel));
        n_lr_pixels[0] = l_index;
    }

    if (n_lr_pixels[1] != r_index) {
        *r_pixels = (struct pixel*)xrealloc(*r_pixels,
                                            r_index * sizeof(struct pixel));
        n_lr_pixels[1] = r_index;
    }
}

static void
shard_results_unref(struct gm_rdt_context_impl* ctx,
                    struct node_data* node_data,
                    struct node_shard_results* results)
{
    if (__builtin_expect(--(results->ref) < 1, 0)) {
        if (ctx->verbose) {
            gm_info(ctx->log, "freeing shard results %p, for node %d",
                    results, node_data->id);
        }
        xfree(results);
    }
}

static void
process_node_shards_work_cb(struct thread_state* state,
                            void* user_data)
{
    struct gm_rdt_context_impl* ctx = state->ctx;
    struct process_node_shards_work* process_work =
        (struct process_node_shards_work*)user_data;

    struct node_shard_results* results = process_work->results;
    int n_shards = results->n_shards;

    struct node_data node_data = process_work->node_data;
    int node_depth = id_to_depth(node_data.id);

    /* Make sure only one worker can process the shard results for a node...
     */
    bool taker = false;
    pthread_mutex_lock(&results->check_lock);
    if (!results->taken) {
        taker = true;
        for (int i = 0; i < n_shards; i++) {
            struct node_shard_data* shard_data = &results->data[i];
            if (!shard_data->done) {
                taker = false;
                break;
            }
        }
        if (taker)
            results->taken = true;
    }
    if (ctx->verbose) {
        if (taker) {
            gm_info(ctx->log, "Processing shard results for node %d, depth=%d",
                    node_data.id, node_depth);
        } else {
            gm_info(ctx->log, "Taker check for node %d (exiting, %s)",
                    node_data.id, results->taken ? "already taken" : "not ready");
        }
    }
    pthread_mutex_unlock(&results->check_lock);

    if (!taker) {
        shard_results_unref(ctx, &process_work->node_data, results);
        xfree(process_work);
        return;
    }

    int best_uv = 0;
    int best_threshold = 0;
    int *n_lr_pixels = NULL;
    float best_gain = 0.0;

    // See which shard got the best uvt combination
    for (int i = 0; i < n_shards; i++) {
        struct node_shard_data* shard_data = &results->data[i];

        if (shard_data->best_gain > best_gain) {
            best_gain = shard_data->best_gain;
            best_uv = shard_data->best_uv;
            best_threshold = shard_data->best_threshold;
            n_lr_pixels = shard_data->n_lr_pixels;
        }
    }

    if (!best_gain) {
        gm_info(ctx->log, "Failed to find a UV threshold combo with any gain");
    }

    // Add this node to the tree and possibly add left/ride nodes to the
    // training queue.
    struct node* node = &ctx->tree[node_data.id];
    if (best_gain > 0.f && (node_depth + 1) < ctx->max_depth)
    {
        memcpy(node->uvs, &ctx->uvs[4 * best_uv], sizeof(node->uvs));
        node->t_mm = ctx->thresholds[best_threshold];

        struct pixel* l_pixels;
        struct pixel* r_pixels;

        collect_pixels(ctx, &node_data, node->uvs, node->t_mm,
                       &l_pixels, &r_pixels, n_lr_pixels);

        int id = (2 * node_data.id) + 1;
        struct node_data ldata;
        ldata.id = id;
        ldata.n_pixels = n_lr_pixels[0];
        ldata.pixels = l_pixels;

        struct node_data rdata;
        rdata.id = id + 1;
        rdata.n_pixels = n_lr_pixels[1];
        rdata.pixels = r_pixels;

        pthread_mutex_lock(&ctx->train_queue_lock);
        ctx->train_queue.push_back(ldata);
        ctx->train_queue.push_back(rdata);
        pthread_mutex_unlock(&ctx->train_queue_lock);

        // Mark the node as a continuing node
        node->label_pr_idx = 0;

        if (ctx->verbose)
        {
            gm_info(ctx->log,
                    "  Node (%u)\n"
                    "    Gain: %f\n"
                    "    U: (%f, %f)\n"
                    "    V: (%f, %f)\n"
                    "    T: %f\n"
                    "  Queued left id=%d, right id=%d\n",
                    node_data.id, best_gain,
                    node->uvs[0] / 1000.0f, node->uvs[1] / 1000.0f,
                    node->uvs[2] / 1000.0f, node->uvs[3] / 1000.0f,
                    node->t_mm / 1000.0f,
                    ldata.id,
                    rdata.id);
        }

    }
    else
    {
        float *nhistogram = results->nhistogram;

        // NB: 0 is reserved for non-leaf nodes
        node->label_pr_idx = (ctx->tree_histograms.size() / ctx->n_labels) + 1;
        int len = ctx->tree_histograms.size();
        ctx->tree_histograms.resize(len + ctx->n_labels);
        memcpy(&ctx->tree_histograms[len], nhistogram, ctx->n_labels * sizeof(float));

        if (ctx->verbose)
        {
            pthread_mutex_lock(&ctx->tidy_log_lock);
            gm_info(ctx->log, "  Leaf node (%d)\n", node_data.id);
            for (int i = 0; i < ctx->n_labels; i++) {
                if (nhistogram[i] > 0.f) {
                    gm_info(ctx->log, "    %02d - %f\n", i, nhistogram[i]);
                }
            }
            pthread_mutex_unlock(&ctx->tidy_log_lock);
        }
    }

    // We no longer need the node's pixel data
    xfree(node_data.pixels);
    node_data.pixels = NULL;

    struct thread_depth_metrics_raw *depth_metrics =
        &state->per_depth_metrics[node_depth];
    depth_metrics->n_nodes++;

    ctx->n_nodes_trained++;
    if (ctx->max_nodes && ctx->n_nodes_trained > ctx->max_nodes) {
        if (ctx->verbose) {
            gm_warn(ctx->log, "Interrupting - Maximum number of nodes (%d) reached",
                    ctx->max_nodes);
        }
        interrupt_reason = "Max nodes trained";
        interrupted = true;
    }

    shard_results_unref(ctx, &node_data, results);
    xfree(process_work);
}

static void*
worker_thread_cb(void* userdata)
{
    struct thread_state* state = (struct thread_state*)userdata;
    struct gm_rdt_context_impl* ctx = state->ctx;

    // We don't expect to be asked to process more than this many uvt
    // combos at a time so we can allocate the memory up front...
    int max_uv_combos_per_thread = (ctx->n_uvs + ctx->n_threads/2) / ctx->n_threads;

    state->uvt_lr_histograms_16.reserve(ctx->n_labels *
                                        max_uv_combos_per_thread *
                                        ctx->n_thresholds *
                                        2);
    state->uvt_lr_histograms_32.reserve(ctx->n_labels *
                                        max_uv_combos_per_thread *
                                        ctx->n_thresholds *
                                        2);

    while (1)
    {
        struct work work = {};

        uint64_t idle_start = get_time();

        pthread_mutex_lock(&ctx->work_queue_lock);
        if (!ctx->work_queue.empty()) {
            work = ctx->work_queue.front();
            ctx->work_queue.pop_front();
        } else {

            /* If we reach the point where all threads are waiting for work
             * then we've implicitly finished training...
             */
            if (++ctx->n_idle == ctx->n_threads) {
                gm_info(ctx->log, "All workers idle");
                // Inform all other threads that we are done...
                interrupted = true;
                pthread_cond_broadcast(&ctx->work_queue_changed);
            }

            while (!interrupted) {
                pthread_cond_wait(&ctx->work_queue_changed, &ctx->work_queue_lock);
                if (!ctx->work_queue.empty()) {
                    work = ctx->work_queue.front();
                    ctx->work_queue.pop_front();
                    break;
                }
            }
            ctx->n_idle--;
        }
        pthread_mutex_unlock(&ctx->work_queue_lock);

        uint64_t idle_end = get_time();

        if (interrupted)
            break;

        int depth = work.depth;
        struct thread_depth_metrics_raw* depth_metrics =
            &state->per_depth_metrics[depth];

        /* XXX: There are no strict guarantees about the order that training
         * nodes are scheduled to the work queue so we can't be sure when we've
         * finished processing a particular depth of the tree.
         *
         * Since we currently assume nodes are processed in (approximately)
         * breadth-first order (there can be some reordering as threads race to
         * schedule work) then we can recognise when we start on a new depth
         * and use that as an opportunity to print metrics about the previous
         * level. Technically there may still be some work left for the
         * previous level but the metrics are hopefully representative.  (NB:
         * the metrics that are logged at the very end will be complete so it's
         * also possible to analyze these if there's any doubt).
         */

        if (depth > 0 && depth_metrics->work_time == 0) {
            uint64_t duration = get_time() - ctx->start;
            char buf[16];

            pthread_mutex_lock(&ctx->tidy_log_lock);

            gm_info(ctx->log, "Thread %2d: total %s, started level %d, recent depth metrics:",
                    state->idx,
                    format_duration_s16(duration, buf),
                    depth);

            // Print metrics for last two levels (the details from two levels
            // are more certainly complete than the previous level)...
            for (int i = depth - 1; i >= 0 && i >= depth - 2; i--) {
                struct thread_depth_metrics_raw* prev_depth_metrics =
                    &state->per_depth_metrics[i];

                if (prev_depth_metrics->work_time) {
                    struct thread_depth_metrics_raw* raw =
                        &state->per_depth_metrics[i];
                    struct thread_depth_metrics_report report;

                    calculate_thread_depth_metrics_report(state,
                                                          i,
                                                          0, // no partial work time
                                                          raw,
                                                          &report);
                    log_thread_depth_metrics(ctx, "> ", i, &report);
                }
            }

            pthread_mutex_unlock(&ctx->tidy_log_lock);
        }

        depth_metrics->idle_time += (idle_end - idle_start);

        state->current_work_start = idle_end;
        work.work_cb(state, work.user_data);
        uint64_t work_end = get_time();
        depth_metrics->work_time += (work_end - state->current_work_start);

        while (schedule_node_work(state))
            ;
    }

    return NULL;
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

    // To help with verbose logging so we can log multiple lines together
    // without interleaving with messages between threads.
    pthread_mutex_init(&ctx->tidy_log_lock, NULL);

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

    ctx->n_thresholds = 49;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_thresholds";
    prop.desc = "Number of thresholds to test";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_thresholds;
    prop.int_state.min = 1;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->threshold_range = 0.5;
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

    ctx->uv_range = 0.4;
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

    ctx->debug_post_inference = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "debug_post_inference";
    prop.desc = "Run label inference on all sampled pixels after training";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->debug_post_inference;
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

    ctx->uvt_histograms_mem = 4000000;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "uvt_histograms_mem";
    prop.desc = "Working set memory constraint for UVT combo histograms";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->uvt_histograms_mem;
    prop.int_state.min = 128000;
    prop.int_state.max = 64000000;
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    return (struct gm_rdt_context *)ctx;
}

static void
destroy_training_state(struct gm_rdt_context_impl* ctx)
{
    ctx->uvs.clear();
    ctx->uvs.shrink_to_fit();
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
                     struct node* node,
                     int depth,
                     int id)
{
    JSON_Value* json_node_val = json_value_init_object();
    JSON_Object* json_node = json_object(json_node_val);

    if (ctx->verbose) {
        json_object_set_number(json_node, "id", id);
    }

    if (node->label_pr_idx == 0)
    {
        json_object_set_number(json_node, "t", node->t_mm / 1000.0f);

        JSON_Value* u_val = json_value_init_array();
        JSON_Array* u = json_array(u_val);
        json_array_append_number(u, node->uvs[0] / 1000.0f);
        json_array_append_number(u, node->uvs[1] / 1000.0f);
        json_object_set_value(json_node, "u", u_val);

        JSON_Value* v_val = json_value_init_array();
        JSON_Array* v = json_array(v_val);
        json_array_append_number(v, node->uvs[2] / 1000.0f);
        json_array_append_number(v, node->uvs[3] / 1000.0f);
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
            struct node* left_node = &ctx->tree[left_id];
            int right_id = id * 2 + 2;
            struct node* right_node = &ctx->tree[right_id];

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
               std::vector<node> &tree,
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

    /* Previous trees without a bg_depth property implicitly considered 1000.0
     * meters to be the background depth, but since we can't represent that
     * in millimeters in an int16_t newer trees are trained to treat 33 meters
     * as the background depth...
     */
    json_object_set_number(json_object(rdt), "bg_depth", INT16_MAX / 1000.0);

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
            struct node_data &root_node,
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

        checkpoint = rdt_tree_load_from_json(ctx->log,
                                             js,
                                             true, // allow loading incomplete trees
                                             err);

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

    /* We track the UVT values as integers instead of float while training... */
    for (int i = 0; i < n_reload_nodes; i++) {
        Node float_node = checkpoint->nodes[i];
        struct node fixed_node = {};

        if (float_node.label_pr_idx == 0) {
            fixed_node.uvs[0] = roundf(float_node.uv[0] * 1000.0f);
            fixed_node.uvs[1] = roundf(float_node.uv[1] * 1000.0f);
            fixed_node.uvs[2] = roundf(float_node.uv[2] * 1000.0f);
            fixed_node.uvs[3] = roundf(float_node.uv[3] * 1000.0f);
            fixed_node.t_mm = roundf(float_node.t * 1000.0f);
        }
        fixed_node.label_pr_idx = float_node.label_pr_idx;
        ctx->tree[i] = fixed_node;
    }

    // Navigate the tree to determine any unfinished nodes and the last
    // trained depth
    std::queue<node_data> reload_queue;

    reload_queue.push(root_node);

    while (reload_queue.size())
    {
        struct node_data node_data = reload_queue.front();
        int node_depth = id_to_depth(node_data.id);
        reload_queue.pop();
        struct node* node = &ctx->tree[node_data.id];

        if (node->label_pr_idx == INT_MAX)
        {
            // INT_MAX implies it wasn't trained yet..
            ctx->train_queue.push_back(node_data);
        }
        else if (node->label_pr_idx != 0 && // leaf node
                 node_depth == (checkpoint->header.depth - 1) &&
                 ctx->max_depth > checkpoint->header.depth)
        {
            /* Also need to train leaf nodes in the last level of the reloaded
             * tree if we're training deeper now...
             */
            node->label_pr_idx = INT_MAX;
            ctx->train_queue.push_back(node_data);
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
            ctx->train_queue.push_back(node_data);
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
                struct pixel* l_pixels;
                struct pixel* r_pixels;
                int n_lr_pixels[] = { 0, 0 };
                collect_pixels(ctx, &node_data,
                               node->uvs, node->t_mm,
                               &l_pixels, &r_pixels,
                               n_lr_pixels);

                int id = (2 * node_data.id) + 1;

                struct node_data ldata;
                ldata.id = id;
                ldata.n_pixels = n_lr_pixels[0];
                ldata.pixels = l_pixels;

                struct node_data rdata;
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

/* A histogram of the labels for the root node pixels is useful to help double
 * check they roughly match the relative sizes of the different labels else
 * maybe there was a problem with generating our sample points.
 */
static void
check_root_pixels_histogram(struct gm_rdt_context_impl* ctx,
                            struct node_data* root_node)
{
    gm_info(ctx->log, "Calculating root node pixel histogram");
    ctx->root_pixel_histogram.resize(ctx->n_labels);
    ctx->root_pixel_nhistogram.resize(ctx->n_labels);
    accumulate_pixels_histogram_32(ctx, root_node, ctx->root_pixel_histogram.data());
    int n_root_pixels = 0;
    int n_root_labels = 0;
    normalize_histogram_32(ctx->root_pixel_histogram.data(),
                           ctx->n_labels,
                           ctx->root_pixel_nhistogram.data(),
                           &n_root_pixels,
                           &n_root_labels);
    JSON_Array* labels = json_object_get_array(json_object(ctx->data_meta),
                                               "labels");
    gm_info(ctx->log, "Histogram of root node pixel labels:");
    print_label_histogram(ctx->log,
                          labels,
                          ctx->root_pixel_nhistogram.data());

    JSON_Value* hist_val = json_value_init_array();
    JSON_Array* hist = json_array(hist_val);
    for (int i = 0; i < ctx->n_labels; i++) {
        json_array_append_number(hist, ctx->root_pixel_nhistogram[i]);
    }

    json_object_set_value(json_object(ctx->record), "root_pixels_histogram",
                          hist_val);
}

struct depth_loader
{
    struct gm_rdt_context_impl* ctx;
    uint64_t last_update;
    int full_width;
    int full_height;
    std::vector<half> image_buf;
    struct bounds* body_bounds;
};

static bool
load_depth_buffers_cb(struct gm_data_index* data_index,
                      int index,
                      const char* frame_path,
                      void* user_data,
                      char** err)
{
    struct depth_loader* loader = (struct depth_loader*)user_data;
    struct gm_rdt_context_impl* ctx = loader->ctx;
    int full_width = loader->full_width;
    int full_height = loader->full_height;
    struct bounds bounds = loader->body_bounds[index];
    struct depth_meta depth_meta = ctx->depth_index[index];
    int cropped_x = bounds.min_x;
    int cropped_y = bounds.min_y;
    int cropped_width = depth_meta.width;
    int cropped_height = depth_meta.height;

    const char* top_dir = gm_data_index_get_top_dir(data_index);
    char depth_filename[512];

    xsnprintf(depth_filename, sizeof(depth_filename), "%s/depth/%s.exr",
              top_dir, frame_path);

    void* tmp_buf = loader->image_buf.data();
    IUImageSpec depth_spec = { full_width, full_height, IU_FORMAT_HALF };
    if (iu_read_exr_from_file(depth_filename, &depth_spec,
                              &tmp_buf) != SUCCESS)
    {
        gm_throw(ctx->log, err, "Failed to read image '%s'\n", depth_filename);
        return false;
    }

    half* src = loader->image_buf.data();
    int src_width = loader->full_width;
    int16_t* dest = &ctx->depth_images[depth_meta.pixel_offset];
    for (int y = 0; y < cropped_height; y++) {
        for (int x = 0; x < cropped_width; x++) {
            int src_x = cropped_x + x;
            int src_y = cropped_y + y;
            int src_off = src_y * src_width + src_x;
            int dest_off = y * cropped_width + x;

            float depth_m = src[src_off];

            gm_assert(ctx->log, !std::isnan(depth_m),
                      "Spurious NAN depth value in training frame %s",
                      frame_path);

            gm_assert(ctx->log, !std::isinf(depth_m),
                      "Spurious INF depth value in training frame %s",
                      frame_path);

            if (depth_m >= HUGE_DEPTH)
                dest[dest_off] = INT16_MAX;
            else {
                gm_assert(ctx->log, depth_m < (INT16_MAX / 1000.0),
                          "Depth value %f in training frame %s can't be represented in 16 bits",
                          depth_m, frame_path);

                dest[dest_off] = depth_m * 1000.0f + 0.5f; // round nearest with the +0.5
            }
        }
    }

    uint64_t current = get_time();
    if (current - ctx->last_load_update > 2000000000) {
        int percent = index * 100 / ctx->n_images;
        gm_info(ctx->log, "%3d%%", percent);
        ctx->last_load_update = current;
    }

    return true;
}

static bool
load_training_data(struct gm_rdt_context_impl* ctx,
                   const char* data_dir,
                   const char* index_name,
                   char** err)
{
    gm_info(ctx->log, "Opening training data index %s...", index_name);
    struct gm_data_index* data_index =
        gm_data_index_open(ctx->log,
                           data_dir,
                           index_name,
                           err);
    if (!data_index)
        return false;

    ctx->data_meta =
        json_value_deep_copy(gm_data_index_get_meta(data_index));

    ctx->n_images = gm_data_index_get_len(data_index);

    JSON_Object* meta_camera =
        json_object_get_object(json_object(ctx->data_meta), "camera");
    ctx->fov = json_object_get_number(meta_camera, "vertical_fov");
    ctx->fov *= (M_PI / 180.0);

    ctx->n_labels = json_object_get_number(json_object(ctx->data_meta), "n_labels");

    gm_assert(ctx->log, ctx->n_labels <= MAX_LABELS,
              "Can't handle training with more than %d labels",
              MAX_LABELS);
    gm_assert(ctx->log, (uint64_t)ctx->n_pixels * ctx->n_images < INT_MAX,
              "Can't handle training with more than %d pixels, but n_pixels * n_images = %" PRIu64,
              INT_MAX, (uint64_t)ctx->n_pixels * ctx->n_images);

    // Create the randomized sample points across all images that the decision
    // tree is going to learn to classify, and associate with a root node...
    //
    // The training recursively splits the pixels at each node of the tree,
    // either terminating when a branch runs out of pixels to differentiate
    // or after reaching the maximum training depth.
    //
    struct node_data root_node;
    root_node.id = 0;
    root_node.pixels = (struct pixel*)xmalloc((size_t)ctx->n_images *
                                              ctx->n_pixels *
                                              sizeof(struct pixel));
    root_node.n_pixels = ctx->n_images * ctx->n_pixels;

    struct bounds *body_bounds = (struct bounds*)xmalloc(ctx->n_images *
                                                         sizeof(struct bounds));
    if (!pre_process_label_images(ctx,
                                  data_index,
                                  root_node.pixels,
                                  body_bounds,
                                  err))
    {
        xfree(root_node.pixels);
        xfree(body_bounds);
        return false;
    }

    ctx->depth_index.resize(ctx->n_images);

    int max_width = gm_data_index_get_width(data_index);
    int max_height = gm_data_index_get_height(data_index);
    int64_t n_depth_pixels = 0;

    for (int i = 0; i < ctx->n_images; i++) {
        struct bounds bounds = body_bounds[i];

        int width = bounds.max_x - bounds.min_x + 1;
        int height = bounds.max_y - bounds.min_y + 1;

        gm_assert(ctx->log, width <= max_width,
                  "Bounded width (%d) > full width (%d)", width, max_width);
        gm_assert(ctx->log, height <= max_height,
                  "Bounded height (%d) > full height (%d)", height, max_height);

        struct depth_meta meta;
        meta.width = width;
        meta.height = height;
        meta.pixel_offset = n_depth_pixels;
        ctx->depth_index[i] = meta;

        n_depth_pixels += (width * height);
    }

    int64_t depth_size = n_depth_pixels * 2;
    int64_t max_depth_size = (int64_t)max_width * max_height * ctx->n_images * 2;
    gm_info(ctx->log, "Size of cropped depth data = %" PRIu64 " bytes, reduced from %" PRIu64 " (%d%% of original size)",
            (int64_t)depth_size,
            (int64_t)max_depth_size,
            (int)((depth_size * 100 / max_depth_size)));

    ctx->depth_images = (int16_t*)xmalloc(depth_size);

    struct depth_loader loader;
    loader.ctx = ctx;
    loader.last_update = get_time();
    loader.full_width = max_width;
    loader.full_height = max_height;
    loader.image_buf = std::vector<half>(max_width * max_height);
    loader.body_bounds = body_bounds;

    gm_info(ctx->log, "Loading all depth buffers...");
    if (!gm_data_index_foreach(data_index,
                               load_depth_buffers_cb,
                               &loader,
                               err))
    {
        xfree(root_node.pixels);
        xfree(body_bounds);
        return false;
    }

    gm_data_index_destroy(data_index);
    data_index = NULL;

    xfree(body_bounds);
    body_bounds = NULL;

    check_root_pixels_histogram(ctx, &root_node);

    // Allocate memory to store the decision tree.
    int n_tree_nodes = (1<<ctx->max_depth) - 1;
    ctx->tree.resize(n_tree_nodes);

    // Mark nodes in tree as unfinished, for checkpoint restoration
    // Note: we still do this if we are reloading a tree, since it may
    // be shallower than the total tree size.
    for (int i = 0; i < (int)ctx->tree.size(); i++) {
        ctx->tree[i].label_pr_idx = INT_MAX;
    }

    if (ctx->reload) {
        if (!reload_tree(ctx, ctx->reload, root_node, err)) {
            xfree(root_node.pixels);
            return false;
        }
    } else {
        ctx->train_queue.push_back(root_node);
    }

    return true;
}

/* Perform label inference in terms of our fixed point sampling code to help
 * spot any discrepancies with other runtime inference implementation
 */
static void
debug_infer_pixel_label(struct gm_rdt_context_impl* ctx,
                        int16_t* depth_image,
                        int width,
                        int height,
                        struct pixel px,
                        float* pr_table_out)
{
    int16_t depth_mm = depth_image[px.y * width + px.x];
    int16_t half_depth_mm = depth_mm / 2;

    int id = 0;
    struct node node = ctx->tree[0];
    while (node.label_pr_idx == 0) {
        int16_t gradient = sample_uv_gradient_mm(depth_image,
                                                 width,
                                                 height,
                                                 px.x, px.y,
                                                 depth_mm,
                                                 half_depth_mm,
                                                 node.uvs);

        /* NB: The nodes are arranged in breadth-first, left then
         * right child order with the root node at index zero.
         *
         * In this case if you have an index for any particular node
         * ('id' here) then 2 * id + 1 is the index for the left
         * child and 2 * id + 2 is the index for the right child...
         */
        id = (gradient < node.t_mm) ? 2 * id + 1 : 2 * id + 2;

        node = ctx->tree[id];
    }

    /* NB: node->label_pr_idx is a base-one index since index zero
     * is reserved to indicate that the node is not a leaf node
     */
    float* pr_table = &ctx->tree_histograms[(node.label_pr_idx - 1) *
        ctx->n_labels];

    memcpy(pr_table_out, pr_table, sizeof(float) * ctx->n_labels);
}

static bool
debug_check_inference(struct gm_rdt_context_impl* ctx,
                      const char* data_dir,
                      const char* index_name,
                      char** err)
{
    int n_labels = ctx->n_labels;

    gm_info(ctx->log, "Re-opening training data index %s...", index_name);
    struct gm_data_index* data_index =
        gm_data_index_open(ctx->log,
                           data_dir,
                           index_name,
                           err);
    if (!data_index)
        return false;

    /* Re-sample the original root node pixels so we can then run label
     * inference on those original pixels
     */
    struct node_data root_node;
    root_node.id = 0;
    root_node.pixels = (struct pixel*)xmalloc((size_t)ctx->n_images *
                                              ctx->n_pixels *
                                              sizeof(struct pixel));
    root_node.n_pixels = ctx->n_images * ctx->n_pixels;

    struct bounds *body_bounds = (struct bounds*)xmalloc(ctx->n_images *
                                                         sizeof(struct bounds));
    if (!pre_process_label_images(ctx,
                                  data_index,
                                  root_node.pixels,
                                  body_bounds,
                                  err))
    {
        xfree(root_node.pixels);
        xfree(body_bounds);
        return false;
    }

    ctx->depth_index.resize(ctx->n_images);

    int max_width = gm_data_index_get_width(data_index);
    int max_height = gm_data_index_get_height(data_index);
    int64_t n_depth_pixels = 0;

    for (int i = 0; i < ctx->n_images; i++) {
        struct bounds bounds = body_bounds[i];

        int width = bounds.max_x - bounds.min_x + 1;
        int height = bounds.max_y - bounds.min_y + 1;

        gm_assert(ctx->log, width <= max_width,
                  "Bounded width (%d) > full width (%d)", width, max_width);
        gm_assert(ctx->log, height <= max_height,
                  "Bounded height (%d) > full height (%d)", height, max_height);

        struct depth_meta meta;
        meta.width = width;
        meta.height = height;
        meta.pixel_offset = n_depth_pixels;
        ctx->depth_index[i] = meta;

        n_depth_pixels += (width * height);
    }

    const char* top_dir = gm_data_index_get_top_dir(data_index);

    for (int i = 0; i < ctx->n_images; i++) {
        int image_pixel_off = ctx->n_pixels * i;
        struct pixel px0 = root_node.pixels[image_pixel_off];

        gm_assert(ctx->log, px0.i == i, "Inconsistent pixel indices after reload");

        struct bounds bounds = body_bounds[i];

        int bounds_width = bounds.max_x - bounds.min_x + 1;
        int bounds_height = bounds.max_y - bounds.min_y + 1;

        struct depth_meta depth_meta = ctx->depth_index[i];

        gm_assert(ctx->log,
                  (bounds_width == depth_meta.width &&
                   bounds_height == depth_meta.height),
                  "Inconsistent bounds and depth_meta for image %d after reload",
                  i);

        char out_filename[512];
        xsnprintf(out_filename, sizeof(out_filename), "%s/labels/%s-training-check.png",
                  top_dir, gm_data_index_get_frame_path(data_index, i));

        uint8_t out[max_width * max_height];
        memset(out, 0, sizeof(out));

        int16_t* depth_image = &ctx->depth_images[depth_meta.pixel_offset];

        for (int p = 0; p < ctx->n_pixels; p++) {
            struct pixel px = root_node.pixels[image_pixel_off + p];
            float pr_table[n_labels];

            debug_infer_pixel_label(ctx,
                                    depth_image,
                                    depth_meta.width,
                                    depth_meta.height,
                                    px,
                                    pr_table);

            float best_prob = pr_table[0];
            int best_l = 0;
            for (int l = 1; l < n_labels; l++) {
                float prob = pr_table[l];
                if (prob > best_prob) {
                    best_prob = prob;
                    best_l = l;
                }
            }

            int out_x = px.x + bounds.min_x;
            int out_y = px.y + bounds.min_y;
            out[max_width * out_y + out_x] = best_l;
        }

        int palette_size = ARRAY_LEN(png_label_palette);
        IUImageSpec output_spec = { max_width, max_height, IU_FORMAT_U8 };
        if (iu_write_png_to_file(out_filename, &output_spec,
                                 out, png_label_palette, palette_size) != SUCCESS)
        {
            gm_error(ctx->log, "Error writing debug PNG %s",
                     out_filename);
        }
        gm_info(ctx->log, "Wrote %s", out_filename);
    }

    gm_data_index_destroy(data_index);
    data_index = NULL;

    return true;
}

static int
meter_range_to_pixelmillimeters(float fov_rad, int res_px, float meter_range)
{
    float field_size_at_1m = 2.0f * tanf(fov_rad / 2.0f);
    float px_per_meter = (float)res_px / field_size_at_1m;

    // + 0.5 to round nearest
    return meter_range * px_per_meter * 1000.0f + 0.5f;
}

bool
gm_rdt_context_train(struct gm_rdt_context* _ctx, char** err)
{
    struct gm_rdt_context_impl* ctx = (struct gm_rdt_context_impl*)_ctx;
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

    /* Loads label data, depth data and potentially loads a pre-existing
     * decision tree...
     */
    if (!load_training_data(ctx, data_dir, index_name, err)) {
        destroy_training_state(ctx);
        return false;
    }

    ctx->record = create_training_record(ctx);

    // Adjust uv range into pixel-millimeters, considering that our depth
    // values are in mm, and we divide uv offsets by the depth to give us depth
    // invariance for uv offsets.
    JSON_Object* camera = json_object_get_object(json_object(ctx->data_meta), "camera");
    int camera_height = json_object_get_number(camera, "height");
    int16_t uv_range_pmm = meter_range_to_pixelmillimeters(ctx->fov,
                                                           camera_height,
                                                           ctx->uv_range);
    gm_info(ctx->log, "UV range = %.2fm = %d pixel-millimeters",
            ctx->uv_range, uv_range_pmm);

    // Calculate the u,v,t parameters that we're going to test
    gm_info(ctx->log, "Preparing training metadata...\n");
    ctx->uvs.resize(ctx->n_uvs * 4);
    std::mt19937 rng(ctx->seed);
    std::uniform_real_distribution<float> rand_uv(-uv_range_pmm / 2,
                                                  uv_range_pmm / 2);
    for (int i = 0; i < ctx->n_uvs * 4; i++)
        ctx->uvs[i] = round(rand_uv(rng));

    if (ctx->n_thresholds % 2 == 0) {
        gm_info(ctx->log, "Increasing N thresholds from %d to %d for symmetry around zero",
                ctx->n_thresholds, ctx->n_thresholds + 1);
        ctx->n_thresholds++;
    }

    ctx->thresholds = (int16_t*)xmalloc(ctx->n_thresholds * sizeof(int16_t));

    int threshold_step_mm = ctx->threshold_range * 500.0 /
        ((ctx->n_thresholds - 1) / 2);

    for (int n = 0; n < ctx->n_thresholds; n++) {
        ctx->thresholds[n] = nth_threshold(n, threshold_step_mm);
        //gm_info(ctx->log, "threshold: %d", ctx->thresholds[n]);
    }

    gm_info(ctx->log, "Initialising %u threads...\n", n_threads);
    ctx->thread_pool.resize(n_threads);

    for (int i = 0; i < n_threads; i++) {
        struct thread_state *state = &ctx->thread_pool[i];
        state->idx = i;
        state->ctx = ctx;
        state->last_metrics_log = get_time();
        state->per_depth_metrics.resize(ctx->max_depth);
    }

    /* This thread will effectively become thread 0 ... */
    ctx->thread_pool[0].thread = pthread_self();
    for (int i = 1; i < n_threads; i++) {
        struct thread_state *state = &ctx->thread_pool[i];

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
    ctx->start = get_time();

    while (schedule_node_work(&ctx->thread_pool[0]))
        ;
    worker_thread_cb(&ctx->thread_pool[0]);

    // NB: thread 0 is this thread...
    for (int i = 1; i < n_threads; i++) {
        struct thread_state *state = &ctx->thread_pool[i];

        if (pthread_join(state->thread, NULL) != 0) {
            gm_error(ctx->log, "Error joining thread, trying to continue...\n");
        }
    }

    JSON_Value *js_metrics = json_value_init_object();
    json_object_set_value(json_object(ctx->record), "metrics", js_metrics);
    JSON_Value *js_dmetrics = NULL; // per-depth metrics

    for (int i = 0; i < ctx->max_depth; i++) {

        if (!js_dmetrics) {
            js_dmetrics = json_value_init_array();
            json_object_set_value(json_object(js_metrics), "per_depth", js_dmetrics);
        }

        JSON_Value *js_depth = json_value_init_object();
        json_object_set_number(json_object(js_depth), "depth", i);
        json_array_append_value(json_array(js_dmetrics), js_depth);

        JSON_Value *js_per_thread = json_value_init_array();
        json_object_set_value(json_object(js_depth), "per_thread", js_per_thread);

        for (int t = 0; t < n_threads; t++) {
            struct thread_state *state = &ctx->thread_pool[t];
            struct thread_depth_metrics_raw *tmetrics = &state->per_depth_metrics[i];
            struct thread_depth_metrics_report treport;

            calculate_thread_depth_metrics_report(state,
                                                  i, // depth
                                                  0, // no partial work time
                                                  tmetrics,
                                                  &treport);
            JSON_Value *js_tmetrics = thread_metrics_to_json(ctx, &treport);
            json_array_append_value(json_array(js_per_thread), js_tmetrics);
        }
    }

    // Write to file
    uint64_t duration = get_time() - ctx->start;
    char buf[16];

    gm_info(ctx->log, "(%s) Writing output to '%s'...\n",
            format_duration_s16(duration, buf),
            out_filename);

    save_tree_json(ctx,
                   ctx->tree,
                   ctx->tree_histograms,
                   out_filename);

    duration = get_time() - ctx->start;
    gm_info(ctx->log, "(%s) %s\n",
            format_duration_s16(duration, buf),
            interrupt_reason ?: "Done!");

    if (ctx->debug_post_inference) {
        char* catch_err = NULL;
        if (!debug_check_inference(ctx, data_dir, index_name, &catch_err)) {
            gm_warn(ctx->log, "Failed to check inference after training: %s",
                    catch_err);
            xfree(catch_err);
        }
    }

    // Free memory that isn't needed anymore
    destroy_training_state(ctx);

    return true;
}
