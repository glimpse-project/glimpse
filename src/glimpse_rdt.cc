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
#include <random>
#include <thread>
#include <pthread.h>
#include <time.h>
#include <signal.h>

#include "half.hpp"

#include "xalloc.h"
#include "llist.h"
#include "utils.h"
#include "loader.h"
#include "train_utils.h"

#include "glimpse_rdt.h"
#include "glimpse_log.h"
#include "glimpse_config.h"

using half_float::half;

static bool interrupted = false;

typedef struct gm_rdt_context_impl {

    struct gm_rdt_context base;

    struct gm_logger* log;

    bool     reload;        // Reload and continue training with pre-existing tree
    bool     verbose;       // Verbose logging
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

    int      n_uv;          // Number of combinations of u,v pairs
    float    uv_range;      // Range of u,v combinations to generate
    int      n_t;           // The number of thresholds
    float    t_range;       // Range of thresholds to test
    int      max_depth;     // Maximum depth to train to
    int      n_pixels;      // Number of pixels to sample
    UVPair*  uvs;           // A list of uv pairs to test
    float*   ts;            // A list of thresholds to test

    int      n_threads;     // How many threads to spawn for training
    int      bg_label;      // label that represents the background

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
};

typedef struct {
    int       id;              // Unique id to place the node a tree.
    int       depth;           // Tree depth at which this node sits.
    int       n_pixels;        // Number of pixels that have reached this node.
    Int3D*    pixels;          // A list of pixel pairs and image indices.
} NodeTrainData;

typedef struct {
    struct gm_rdt_context_impl* ctx;       // The context to use
    NodeTrainData**    data;               // The node data to use and modify
    int                c_start;            // The uv combination to start on
    int                c_end;              // The uv combination to end on
    float*             root_nhistogram;    // Normalised histogram of labels
    float*             best_gain;          // Best gain achieved
    int*               best_uv;            // Index of the best uv combination
    int*               best_t;             // Index of the best threshold
    int*               n_lr_pixels;        // Number of pixels in each branch
    pthread_barrier_t* ready_barrier;      // Barrier to wait on to start work
    pthread_barrier_t* finished_barrier;   // Barrier to wait on when finished
} TrainThreadData;

static NodeTrainData*
create_node_train_data(struct gm_rdt_context_impl* ctx, int id, int depth,
                       int n_pixels, Int3D* pixels)
{
    NodeTrainData* data = (NodeTrainData*)xcalloc(1, sizeof(NodeTrainData));

    data->id = id;
    data->depth = depth;

    if (pixels)
    {
        data->pixels = pixels;
        data->n_pixels = n_pixels;
    }
    else
    {
        // Assume this is the root node and generate random coordinates
        data->n_pixels = ctx->n_images * ctx->n_pixels;
        data->pixels = (Int3D*)xmalloc(data->n_pixels * sizeof(Int3D));

        //std::random_device rd;
        std::mt19937 rng(ctx->seed);
        std::uniform_int_distribution<int> rand_x(0, ctx->width - 1);
        std::uniform_int_distribution<int> rand_y(0, ctx->height - 1);
        for (int i = 0, idx = 0; i < ctx->n_images; i++)
        {
            for (int j = 0; j < ctx->n_pixels; j++, idx++)
            {
                data->pixels[idx].xy[0] = rand_x(rng);
                data->pixels[idx].xy[1] = rand_y(rng);
                data->pixels[idx].i = i;
            }
        }
    }

    return data;
}

static void
destroy_node_train_data(NodeTrainData* data)
{
    xfree(data->pixels);
    xfree(data);
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
accumulate_histograms(struct gm_rdt_context_impl* ctx,
                      NodeTrainData* data,
                      int c_start, int c_end,
                      int* root_histogram, int* lr_histograms)
{
    for (int p = 0; p < data->n_pixels && !interrupted; p++)
    {
        Int2D pixel = data->pixels[p].xy;
        int i = data->pixels[p].i;
        int image_idx = i * ctx->width * ctx->height;

        half* depth_image = &ctx->depth_images[image_idx];
        uint8_t* label_image = &ctx->label_images[image_idx];

        int pixel_idx = (pixel[1] * ctx->width) + pixel[0];
        int label = (int)label_image[pixel_idx];
        float depth = depth_image[pixel_idx];

        gm_assert(ctx->log, label < ctx->n_labels,
                  "Label '%d' is bigger than expected (max %d)\n",
                  label, ctx->n_labels - 1);

        // Accumulate root histogram
        ++root_histogram[label];

        // Don't waste processing time if this is the last depth
        if (data->depth >= ctx->max_depth - 1) {
            continue;
        }

        // Accumulate LR branch histograms

        // Sample pixels
        float samples[c_end - c_start];
        for (int c = c_start; c < c_end; c++)
        {
            UVPair uv = ctx->uvs[c];
            samples[c - c_start] = sample_uv(depth_image,
                                             ctx->width, ctx->height,
                                             pixel, depth, uv);
        }

        // Partition on thresholds
        for (int c = 0, lr_histogram_idx = 0; c < c_end - c_start; c++)
        {
            for (int t = 0; t < ctx->n_t;
                 t++, lr_histogram_idx += ctx->n_labels * 2)
            {
                // Accumulate histogram for this particular uvt combination
                // on both theoretical branches
                float threshold = ctx->ts[t];
                ++lr_histograms[samples[c] < threshold ?
                    lr_histogram_idx + label :
                    lr_histogram_idx + ctx->n_labels + label];
            }
        }
    }
}

static void*
thread_body(void* userdata)
{
    TrainThreadData* data = (TrainThreadData*)userdata;

    // Histogram for the node being processed
    int* root_histogram = (int*)
        malloc(data->ctx->n_labels * sizeof(int));

    // Histograms for each uvt combination being tested
    int* lr_histograms = (int*)
        malloc(data->ctx->n_labels * (data->c_end - data->c_start) *
               data->ctx->n_t * 2 * sizeof(int));

    float* nhistogram = (float*)xmalloc(data->ctx->n_labels * sizeof(float));
    float* root_nhistogram = data->root_nhistogram ? data->root_nhistogram :
        (float*)xmalloc(data->ctx->n_labels * sizeof(float));

    while (1)
    {
        // Wait for everything to be ready to start processing
        pthread_barrier_wait(data->ready_barrier);

        // Quit out if we've nothing left to process or we've been interrupted
        if (!(*data->data) || interrupted)
        {
            break;
        }

        // Clear histogram accumulators
        memset(root_histogram, 0, data->ctx->n_labels * sizeof(int));
        memset(lr_histograms, 0, data->ctx->n_labels *
               (data->c_end - data->c_start) * data->ctx->n_t * 2 *
               sizeof(int));

        // Accumulate histograms
        accumulate_histograms(data->ctx, *data->data, data->c_start, data->c_end,
                              root_histogram, lr_histograms);

        // Calculate the normalised label histogram and get the number of pixels
        // and the number of labels in the root histogram.
        Int2D root_n_pixels = normalize_histogram(root_histogram,
                                                  data->ctx->n_labels,
                                                  root_nhistogram);

        // Determine the best u,v,t combination
        *data->best_gain = 0.f;

        // If there's only 1 label, skip all this, gain is zero
        if (root_n_pixels[1] > 1 &&
            (*data->data)->depth < data->ctx->max_depth - 1)
        {
            // Calculate the shannon entropy for the normalised label histogram
            float entropy = calculate_shannon_entropy(root_nhistogram,
                                                      data->ctx->n_labels);

            // Calculate the gain for each combination of u,v,t and store the best
            for (int i = data->c_start, lr_histo_base = 0;
                 i < data->c_end && !interrupted; i++)
            {
                for (int j = 0; j < data->ctx->n_t && !interrupted;
                     j++, lr_histo_base += data->ctx->n_labels * 2)
                {
                    float l_entropy, r_entropy, gain;

                    Int2D l_n_pixels =
                        normalize_histogram(&lr_histograms[lr_histo_base],
                                            data->ctx->n_labels, nhistogram);
                    if (l_n_pixels[0] == 0 || l_n_pixels[0] == root_n_pixels[0])
                    {
                        continue;
                    }
                    l_entropy = calculate_shannon_entropy(nhistogram,
                                                          data->ctx->n_labels);

                    Int2D r_n_pixels =
                        normalize_histogram(
                            &lr_histograms[lr_histo_base + data->ctx->n_labels],
                            data->ctx->n_labels, nhistogram);
                    r_entropy = calculate_shannon_entropy(nhistogram,
                                                          data->ctx->n_labels);

                    gain = calculate_gain(entropy, root_n_pixels[0],
                                          l_entropy, l_n_pixels[0],
                                          r_entropy, r_n_pixels[0]);

                    if (gain > *data->best_gain)
                    {
                        *data->best_gain = gain;
                        *data->best_uv = i;
                        *data->best_t = j;
                        data->n_lr_pixels[0] = l_n_pixels[0];
                        data->n_lr_pixels[1] = r_n_pixels[0];
                    }
                }
            }
        }

        // Signal work is finished
        pthread_barrier_wait(data->finished_barrier);
    }

    xfree(root_histogram);
    xfree(lr_histograms);
    if (!data->root_nhistogram)
    {
        xfree(root_nhistogram);
    }
    xfree(nhistogram);
    xfree(data);

    pthread_exit(NULL);
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
        int image_idx = pixel->i * ctx->width * ctx->height;
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

static bool
list_free_cb(LList* node, uint32_t index, void* userdata)
{
    xfree(node->data);
    return true;
}

void
sigint_handler(int signum)
{
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

    ctx->reload = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "reload";
    prop.desc = "Reload and continue training pre-existing tree";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->reload;
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

    ctx->n_t = 50;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_t";
    prop.desc = "Number of thresholds to test";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_t;
    prop.int_state.min = 1;
    prop.int_state.max = INT_MAX;
    ctx->properties.push_back(prop);

    ctx->t_range = 1.29;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "t_range";
    prop.desc = "Range of thresholds to test";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->t_range;
    prop.float_state.min = 0;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->n_uv = 2000;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "n_uv";
    prop.desc = "Number of UV combinations to test";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->n_uv;
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
    prop.int_state.min = 1;
    prop.int_state.max = 30;
    ctx->properties.push_back(prop);

    ctx->bg_label = 0;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bg_label";
    prop.desc = "Index of the background label";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->bg_label;
    prop.int_state.min = 0;
    prop.int_state.max = 255;
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

    ctx->verbose = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "verbose";
    prop.desc = "Verbose logging output";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->verbose;
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

    return &ctx->base;
}

void
gm_rdt_context_destroy(struct gm_rdt_context *_ctx)
{
    struct gm_rdt_context_impl *ctx = (struct gm_rdt_context_impl *)_ctx;
    gm_logger_destroy(ctx->log);
    delete ctx;
}

bool
gm_rdt_context_train(struct gm_rdt_context *_ctx, char **err)
{
    struct gm_rdt_context_impl *ctx = (struct gm_rdt_context_impl *)_ctx;
    TimeForDisplay since_begin, since_last;
    struct timespec begin, last, now;
    int n_threads = ctx->n_threads;

    const char *data_dir = ctx->data_dir;
    if (!data_dir) {
        gm_throw(ctx->log, err, "Data directory not specified");
        return false;
    }
    const char *index_name = ctx->index_name;
    if (!index_name) {
        gm_throw(ctx->log, err, "Index name not specified");
        return false;
    }
    const char *out_filename = ctx->out_filename;
    if (!out_filename) {
        gm_throw(ctx->log, err, "Output filename not specified");
        return false;
    }

    gm_info(ctx->log, "Scanning training directories...\n");
    gather_train_data(data_dir,
                      index_name,
                      NULL,     // no joint map
                      &ctx->n_images, NULL, &ctx->width, &ctx->height,
                      &ctx->depth_images, &ctx->label_images, NULL,
                      &ctx->n_labels,
                      &ctx->fov);

    // Work out pixels per meter and adjust uv range accordingly
    float ppm = (ctx->height / 2.f) / tanf(ctx->fov / 2.f);
    ctx->uv_range *= ppm;

    // Calculate the u,v,t parameters that we're going to test
    gm_info(ctx->log, "Preparing training metadata...\n");
    ctx->uvs = (UVPair*)xmalloc(ctx->n_uv * sizeof(UVPair));
    //std::random_device rd;
    std::mt19937 rng(ctx->seed);
    std::uniform_real_distribution<float> rand_uv(-ctx->uv_range / 2.f,
                                                  ctx->uv_range / 2.f);
    for (int i = 0; i < ctx->n_uv; i++) {
        ctx->uvs[i][0] = rand_uv(rng);
        ctx->uvs[i][1] = rand_uv(rng);
        ctx->uvs[i][2] = rand_uv(rng);
        ctx->uvs[i][3] = rand_uv(rng);
    }
    ctx->ts = (float*)xmalloc(ctx->n_t * sizeof(float));
    for (int i = 0; i < ctx->n_t; i++) {
        ctx->ts[i] = -ctx->t_range / 2.f + (i * ctx->t_range / (float)(ctx->n_t - 1));
    }

    // Allocate memory for the normalised histogram of the currently training node
    float* root_nhistogram = (float*)xmalloc(ctx->n_labels * sizeof(float));

    NodeTrainData* node_data = NULL;
    gm_info(ctx->log, "Initialising %u threads...\n", n_threads);
    pthread_barrier_t ready_barrier, finished_barrier;
    if (pthread_barrier_init(&ready_barrier, NULL, n_threads + 1) != 0 ||
        pthread_barrier_init(&finished_barrier, NULL, n_threads + 1) != 0)
    {
        gm_throw(ctx->log, err, "Error initialising thread barriers\n");
        return false;
    }
    int n_c = ctx->n_uv / n_threads;
    float* best_gains = (float*)malloc(n_threads * sizeof(float));
    int* best_uvs = (int*)malloc(n_threads * sizeof(int));
    int* best_ts = (int*)malloc(n_threads * sizeof(int));
    int* all_n_lr_pixels = (int*)malloc(n_threads * 2 * sizeof(int));
    pthread_t threads[n_threads];
    for (int i = 0; i < n_threads; i++)
    {
        TrainThreadData* thread_data = (TrainThreadData*)
            xmalloc(sizeof(TrainThreadData));
        thread_data->ctx = ctx;
        thread_data->data = &node_data;
        thread_data->c_start = i * n_c;
        thread_data->c_end = (i == n_threads - 1) ? ctx->n_uv : (i + 1) * n_c;
        thread_data->root_nhistogram = (i == 0) ? root_nhistogram : NULL;
        thread_data->best_gain = &best_gains[i];
        thread_data->best_uv = &best_uvs[i];
        thread_data->best_t = &best_ts[i];
        thread_data->n_lr_pixels = &all_n_lr_pixels[i * 2];
        thread_data->ready_barrier = &ready_barrier;
        thread_data->finished_barrier = &finished_barrier;

        if (pthread_create(&threads[i], NULL, thread_body,
                           (void*)thread_data) != 0)
        {
            gm_throw(ctx->log, err, "Error creating thread\n");
            return false;
        }
    }

    // Allocate memory to store the decision tree.
    int n_nodes = roundf(powf(2.f, ctx->max_depth)) - 1;
    Node* tree = (Node*)xcalloc(n_nodes, sizeof(Node));

    // Initialise root node training data and add it to the queue
    LList* train_queue = llist_new(create_node_train_data(ctx, 0, 0, 0, NULL));

    // Initialise histogram count
    int n_histograms = 0;
    LList* tree_histograms = NULL;

    // If asked to reload then try to load the partial tree and repopulate the
    // training queue and tree histogram list
    RDTree* checkpoint;
    if (ctx->reload && (checkpoint = read_tree(out_filename)))
    {
        gm_info(ctx->log, "Restoring checkpoint...\n");

        // Do some basic validation
        if (checkpoint->header.n_labels != ctx->n_labels)
        {
            gm_throw(ctx->log, err, "Checkpoint has %d labels, expected %d\n",
                     (int)checkpoint->header.n_labels, ctx->n_labels);
            return false;
        }

        if (fabs(checkpoint->header.fov - ctx->fov) > 1e-6)
        {
            gm_throw(ctx->log, err, "Checkpoint has FOV %.2f, expected %.2f\n",
                     checkpoint->header.fov, ctx->fov);
            return false;
        }

        if (checkpoint->header.depth > ctx->max_depth)
        {
            gm_throw(ctx->log, err,
                     "Can't train with a lower depth than checkpoint (%d < %d)\n",
                     ctx->max_depth, (int)checkpoint->header.depth);
            return false;
        }

        // Restore nodes
        int n_checkpoint_nodes = roundf(powf(2.f, checkpoint->header.depth)) - 1;
        memcpy(tree, checkpoint->nodes, n_checkpoint_nodes * sizeof(Node));

        // Navigate the tree to determine any unfinished nodes and the last
        // trained depth
        LList* checkpoint_queue = train_queue;
        train_queue = NULL;
        while (checkpoint_queue)
        {
            NodeTrainData* data = (NodeTrainData*)
                llist_pop(&checkpoint_queue, NULL, NULL);
            Node* node = &tree[data->id];

            // Check if the node has a valid probability table and copy it to
            // the list if so. Given the order in which we iterate over the tree,
            // we can just append to the list. Note that the code expects
            // tree_histograms to point to the end of the list.
            if (node->label_pr_idx != 0 && node->label_pr_idx != INT_MAX)
            {
                float* pr_table = &checkpoint->
                    label_pr_tables[ctx->n_labels * (node->label_pr_idx - 1)];
                float* pr_copy = (float*)xmalloc(ctx->n_labels * sizeof(float));
                memcpy(pr_copy, pr_table, ctx->n_labels * sizeof(float));
                tree_histograms = llist_insert_after(tree_histograms,
                                                     llist_new(pr_copy));
                ++n_histograms;
            }

            // Check if the node is either marked as incomplete, or it sits on
            // the last depth of the tree and we're trying to train deeper.
            if (node->label_pr_idx == INT_MAX ||
                (data->depth == (checkpoint->header.depth - 1) &&
                 ctx->max_depth > checkpoint->header.depth))
            {
                // This node is referenced and incomplete, add it to the training
                // queue.
                train_queue ?
                    llist_insert_after(train_queue, llist_new(data)) :
                    train_queue = llist_new(data);
                continue;
            }

            // If the node isn't a leaf-node, calculate which pixels should go
            // to the next two nodes and add them to the checkpoint queue
            if (node->label_pr_idx == 0)
            {
                Int3D* l_pixels;
                Int3D* r_pixels;
                int n_lr_pixels[] = { 0, 0 };
                collect_pixels(ctx, data, node->uv, node->t, &l_pixels, &r_pixels,
                               n_lr_pixels);

                int id = (2 * data->id) + 1;
                int depth = data->depth + 1;
                NodeTrainData* ldata = create_node_train_data(
                    ctx, id, depth, n_lr_pixels[0], l_pixels);
                NodeTrainData* rdata = create_node_train_data(
                    ctx, id + 1, depth, n_lr_pixels[1], r_pixels);

                checkpoint_queue =
                    llist_first(
                        llist_insert_after(
                            llist_insert_after(llist_last(checkpoint_queue),
                                               llist_new(ldata)),
                            llist_new(rdata)));
            }

            // Free the unused training data
            destroy_node_train_data(data);
        }

        free_tree(checkpoint);

        if (!train_queue)
        {
            gm_throw(ctx->log, err, "Tree already fully trained.\n");
            return false;
        }
    }
    else
    {
        // Mark nodes in tree as unfinished, for checkpoint restoration
        for (int i = 0; i < n_nodes; i++) {
            tree[i].label_pr_idx = INT_MAX;
        }
    }

    gm_info(ctx->log, "Beginning training...\n");
    signal(SIGINT, sigint_handler);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    last = begin;
    int last_depth = INT_MAX;
    while (train_queue != NULL)
    {
        int best_uv = 0;
        int best_t = 0;
        int *n_lr_pixels = NULL;
        float best_gain = 0.0;

        LList* current = train_queue;
        node_data = (NodeTrainData*)current->data;

        if (node_data->depth != last_depth)
        {
            clock_gettime(CLOCK_MONOTONIC, &now);
            since_begin = get_time_for_display(&begin, &now);
            since_last = get_time_for_display(&last, &now);
            last = now;
            last_depth = node_data->depth;
            gm_info(ctx->log,
                    "(%02d:%02d:%02d / %02d:%02d:%02d) Training depth %u (%u nodes)\n",
                    since_begin.hours, since_begin.minutes, since_begin.seconds,
                    since_last.hours, since_last.minutes, since_last.seconds,
                    last_depth + 1, llist_length(train_queue));
        }

        // Signal threads to start work
        pthread_barrier_wait(&ready_barrier);

        // Wait for threads to finish
        pthread_barrier_wait(&finished_barrier);

        // Quit if we've been interrupted
        if (interrupted) {
            gm_warn(ctx->log, "Stopping training due to user-triggered interrupt");
            break;
        }

        // See which thread got the best uvt combination
        for (int i = 0; i < n_threads; i++)
        {
            if (best_gains[i] > best_gain)
            {
                best_gain = best_gains[i];
                best_uv = best_uvs[i];
                best_t = best_ts[i];
                n_lr_pixels = &all_n_lr_pixels[i * 2];
            }
        }

        // Add this node to the tree and possible add left/ride nodes to the
        // training queue.
        Node* node = &tree[node_data->id];
        if (best_gain > 0.f && (node_data->depth + 1) < ctx->max_depth)
        {
            node->uv = ctx->uvs[best_uv];
            node->t = ctx->ts[best_t];
            if (ctx->verbose)
            {
                gm_info(ctx->log,
                        "  Node (%u)\n"
                        "    Gain: %f\n"
                        "    U: (%f, %f)\n"
                        "    V: (%f, %f)\n"
                        "    T: %f\n",
                        node_data->id, best_gain,
                        node->uv[0], node->uv[1],
                        node->uv[2], node->uv[3],
                        node->t);
            }

            Int3D* l_pixels;
            Int3D* r_pixels;

            collect_pixels(ctx, node_data, node->uv, node->t,
                           &l_pixels, &r_pixels, n_lr_pixels);

            int id = (2 * node_data->id) + 1;
            int depth = node_data->depth + 1;
            NodeTrainData* ldata = create_node_train_data(
                ctx, id, depth, n_lr_pixels[0], l_pixels);
            NodeTrainData* rdata = create_node_train_data(
                ctx, id + 1, depth, n_lr_pixels[1], r_pixels);

            // Insert nodes into the training queue
            llist_insert_after(
                llist_insert_after(llist_last(train_queue), llist_new(ldata)),
                llist_new(rdata));

            // Mark the node as a continuing node
            node->label_pr_idx = 0;
        }
        else
        {
            if (ctx->verbose)
            {
                gm_info(ctx->log, "  Leaf node (%d)\n", node_data->id);
                for (int i = 0; i < ctx->n_labels; i++) {
                    if (root_nhistogram[i] > 0.f) {
                        gm_info(ctx->log, "    %02d - %f\n", i, root_nhistogram[i]);
                    }
                }
            }

            node->label_pr_idx = ++n_histograms;
            float* node_histogram = (float*)xmalloc(ctx->n_labels * sizeof(float));
            memcpy(node_histogram, root_nhistogram, ctx->n_labels * sizeof(float));
            tree_histograms = llist_insert_after(tree_histograms,
                                                 llist_new(node_histogram));
        }

        // Remove this node from the queue
        train_queue = train_queue->next;
        llist_free(llist_remove(current), NULL, NULL);

        // We no longer need the train data, free it
        destroy_node_train_data(node_data);
    }

    // Signal threads to free memory and quit
    node_data = NULL;
    pthread_barrier_wait(&ready_barrier);

    for (int i = 0; i < n_threads; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            gm_error(ctx->log, "Error joining thread, trying to continue...\n");
        }
    }

    // Free memory that isn't needed anymore
    xfree(root_nhistogram);
    xfree(ctx->uvs);
    xfree(ctx->ts);
    xfree(ctx->label_images);
    xfree(ctx->depth_images);
    xfree(best_gains);
    xfree(best_uvs);
    xfree(best_ts);
    xfree(all_n_lr_pixels);

    // Restore tree histograms list pointer
    tree_histograms = llist_first(tree_histograms);

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

    RDTHeader header = {
        { 'R', 'D', 'T' },
        RDT_VERSION,
        (uint8_t)ctx->max_depth,
        (uint8_t)ctx->n_labels,
        (uint8_t)ctx->bg_label,
        ctx->fov
    };
    RDTree rdtree = { header, tree, llist_length(tree_histograms), NULL };
    rdtree.label_pr_tables = (float*)
        xmalloc(ctx->n_labels * rdtree.n_pr_tables * sizeof(float));
    float* pr_table = rdtree.label_pr_tables;
    for (LList* l = tree_histograms; l; l = l->next, pr_table += ctx->n_labels)
    {
        memcpy(pr_table, l->data, sizeof(float) * ctx->n_labels);
    }

    save_tree(&rdtree, out_filename);

    char* out_filename_json = (char*)xmalloc(strlen(out_filename) + 6);
    strcpy(out_filename_json, out_filename);
    strcat(out_filename_json, ".json");
    save_tree_json(&rdtree, out_filename_json, false);

    xfree(rdtree.label_pr_tables);
    xfree(out_filename_json);

    // Free the last data
    xfree(tree);
    llist_free(tree_histograms, list_free_cb, NULL);

    clock_gettime(CLOCK_MONOTONIC, &now);
    since_begin = get_time_for_display(&begin, &now);
    since_last = get_time_for_display(&last, &now);
    last = now;
    gm_info(ctx->log,
            "(%02d:%02d:%02d / %02d:%02d:%02d) %s\n",
            since_begin.hours, since_begin.minutes, since_begin.seconds,
            since_last.hours, since_last.minutes, since_last.seconds,
            interrupted ? "Interrupted!" : "Done!");

    return true;
}
