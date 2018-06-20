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


#include <stdbool.h>
#include <math.h>
#include <thread>
#include <pthread.h>

#include "half.hpp"

#include "infer_labels.h"
#include "xalloc.h"
#include "rdt_tree.h"


using half_float::half;

typedef struct {
    int thread;
    int n_threads;
    RDTree** forest;
    int n_trees;
    void* depth_image;
    int width;
    int height;
    float* output;
} InferThreadData;

typedef vector(int, 2) Int2D;

/* Used with newer trees that were trained with fixed-point arithmetic when
 * sampling and testing UV gradients...
 */
template<typename FloatT>
static void*
infer_labels_2018_06_discrete_mm(void* userdata)
{
    InferThreadData* data = (InferThreadData*)userdata;
    FloatT* depth_image = (FloatT*)data->depth_image;
    int n_labels = data->forest[0]->header.n_labels;

    /* Bias by 2cm because half-float depth buffers can loose a lot of
     * precision. E.g. (INT16_MAX / 1000.0) which is the bg_depth for new trees
     * reads back as as 32.75 meters instead of 32.767 which has an error of
     * nearly 2cm
     */
    float bg_depth = data->forest[0]->header.bg_depth - 0.02;

    int width = data->width;
    int height = data->height;

    // Accumulate probability map
    for (int off = data->thread;
         off < width * height;
         off += data->n_threads)
    {
        int y = off / data->width;
        int x = off % data->width;

        float* out_pr_table = &data->output[off * n_labels];
        float depth = depth_image[off];

        int16_t depth_mm = depth * 1000.0f + 0.5f;
        int16_t half_depth_mm = depth_mm / 2;

        if (depth >= bg_depth)
        {
            out_pr_table[data->forest[0]->header.bg_label] += 1.0f;
            continue;
        }

        for (int i = 0; i < data->n_trees; ++i)
        {
            RDTree* tree = data->forest[i];
            Node node = tree->nodes[0];

            int id = 0;
            while (node.label_pr_idx == 0) {
#if 0
                Int2D u = { (int)(x + roundf(node.uv[0] / depth)),
                            (int)(y + roundf(node.uv[1] / depth)) };
                Int2D v = { (int)(x + roundf(node.uv[2] / depth)),
                            (int)(y + roundf(node.uv[3] / depth)) };
#if 0
                float upixel = (u[0] >= 0 && u[0] < (int)width &&
                                u[1] >= 0 && u[1] < (int)height) ?
                    (float)depth_image[((u[1] * width) + u[0])] : ((float)INT16_MAX/1000.0f);
                float vpixel = (v[0] >= 0 && v[0] < (int)width &&
                                v[1] >= 0 && v[1] < (int)height) ?
                    (float)depth_image[((v[1] * width) + v[0])] : ((float)INT16_MAX/1000.0f);

                /* XXX: because we measure gradients discretely in millimeters
                 * while training the thresholds can be sensitive to rounding
                 * details and this matches the training arithmetic better
                 * than:
                 *
                 *   float gradient = upixel - vpixel;
                 */
                int16_t u_z = roundf(upixel * 1000.0f);
                int16_t v_z = roundf(vpixel * 1000.0f);
                float gradient = (float)(u_z - v_z) / 1000.0f;

                /* NB: The nodes are arranged in breadth-first, left then
                 * right child order with the root node at index zero.
                 *
                 * In this case if you have an index for any particular node
                 * ('id' here) then 2 * id + 1 is the index for the left
                 * child and 2 * id + 2 is the index for the right child...
                 */
                id = (gradient < node.t) ? 2 * id + 1 : 2 * id + 2;
#else
                int16_t u_z;
                if (u[0] >= 0 && u[0] < (int)width && u[1] >= 0 && u[1] < (int)height)
                    u_z = (float)depth_image[((u[1] * width) + u[0])] * 1000.0f + 0.5f; // round nearest
                else
                    u_z = INT16_MAX;
                int16_t v_z;
                if (v[0] >= 0 && v[0] < (int)width && v[1] >= 0 && v[1] < (int)height)
                    v_z = (float)depth_image[((v[1] * width) + v[0])] * 1000.0f + 0.5f; // round nearest
                else
                    v_z = INT16_MAX;

                int16_t gradient = u_z - v_z;
                int16_t t_mm = roundf(node.t * 1000.0f);

                /* NB: The nodes are arranged in breadth-first, left then
                 * right child order with the root node at index zero.
                 *
                 * In this case if you have an index for any particular node
                 * ('id' here) then 2 * id + 1 is the index for the left
                 * child and 2 * id + 2 is the index for the right child...
                 */
                id = (gradient < t_mm) ? 2 * id + 1 : 2 * id + 2;
#endif

#else

#define div_int_round_nearest(N, D, HALF_D) \
    ((N < 0) ? ((N - HALF_D)/D) : ((N + HALF_D)/D))

                int32_t uvs[4] = {
                    (int32_t)roundf(node.uv[0] * 1000.0f),
                    (int32_t)roundf(node.uv[1] * 1000.0f),
                    (int32_t)roundf(node.uv[2] * 1000.0f),
                    (int32_t)roundf(node.uv[3] * 1000.0f)
                };
                int32_t u_x = x + div_int_round_nearest(uvs[0], depth_mm, half_depth_mm);
                int32_t u_y = y + div_int_round_nearest(uvs[1], depth_mm, half_depth_mm);
                int32_t v_x = x + div_int_round_nearest(uvs[2], depth_mm, half_depth_mm);
                int32_t v_y = y + div_int_round_nearest(uvs[3], depth_mm, half_depth_mm);

                int16_t u_z;
                if (u_x >= 0 && u_x < width && u_y >= 0 && u_y < height)
                    u_z = depth_image[(u_y * width + u_x)] * 1000.0f + 0.5f; // round nearest
                else
                    u_z = INT16_MAX;
                int16_t v_z;
                if (v_x >= 0 && v_x < width && v_y >= 0 && v_y < height)
                    v_z = depth_image[(v_y * width + v_x)] * 1000.0f + 0.5f; // round nearest
                else
                    v_z = INT16_MAX;

                int16_t gradient = u_z - v_z;
                int16_t t_mm = roundf(node.t * 1000.0f);

                /* NB: The nodes are arranged in breadth-first, left then
                 * right child order with the root node at index zero.
                 *
                 * In this case if you have an index for any particular node
                 * ('id' here) then 2 * id + 1 is the index for the left
                 * child and 2 * id + 2 is the index for the right child...
                 */
                id = (gradient < t_mm) ? 2 * id + 1 : 2 * id + 2;
#endif

                node = tree->nodes[id];
            }

            /* NB: node->label_pr_idx is a base-one index since index zero
             * is reserved to indicate that the node is not a leaf node
             */
            float* pr_table =
                &tree->label_pr_tables[(node.label_pr_idx - 1) * n_labels];
            for (int n = 0; n < n_labels; ++n)
            {
                out_pr_table[n] += pr_table[n];
            }
        }

        for (int n = 0; n < n_labels; ++n)
        {
            out_pr_table[n] /= (float)data->n_trees;
        }
    }

    if (data->n_threads > 1)
    {
        pthread_exit(NULL);
    }

    return NULL;
}

/* Use this inference implementation with legacy decision trees that
 * were trained using floor() rounding when normalizing uv offsets
 * and measured gradients in floating point with meter units.
 */
template<typename FloatT>
static void*
infer_labels_2017_floor_uv_float_m(void* userdata)
{
    InferThreadData* data = (InferThreadData*)userdata;
    FloatT* depth_image = (FloatT*)data->depth_image;
    int n_labels = data->forest[0]->header.n_labels;

    /* Bias by 2cm because half-float depth buffers can loose a lot of
     * precision. E.g. (INT16_MAX / 1000.0) which is the bg_depth for new trees
     * reads back as as 32.75 meters instead of 32.767 which has an error of
     * nearly 2cm
     */
    float bg_depth = data->forest[0]->header.bg_depth - 0.02;

    int width = data->width;
    int height = data->height;

    // Accumulate probability map
    for (int off = data->thread;
         off < width * height;
         off += data->n_threads)
    {
        int y = off / data->width;
        int x = off % data->width;

        float* out_pr_table = &data->output[off * n_labels];
        float depth = depth_image[off];

        if (depth >= bg_depth)
        {
            out_pr_table[data->forest[0]->header.bg_label] += 1.0f;
            continue;
        }

        Int2D pixel = { x, y };
        for (int i = 0; i < data->n_trees; ++i)
        {
            RDTree* tree = data->forest[i];
            Node node = tree->nodes[0];

            int id = 0;
            while (node.label_pr_idx == 0) {
                Int2D u = { (int)(pixel[0] + node.uv[0] / depth),
                            (int)(pixel[1] + node.uv[1] / depth) };
                Int2D v = { (int)(pixel[0] + node.uv[2] / depth),
                            (int)(pixel[1] + node.uv[3] / depth) };

                float upixel = (u[0] >= 0 && u[0] < (int)width &&
                                u[1] >= 0 && u[1] < (int)height) ?
                    (float)depth_image[((u[1] * width) + u[0])] : 1000.f;
                float vpixel = (v[0] >= 0 && v[0] < (int)width &&
                                v[1] >= 0 && v[1] < (int)height) ?
                    (float)depth_image[((v[1] * width) + v[0])] : 1000.f;

                float gradient = upixel - vpixel;

                /* NB: The nodes are arranged in breadth-first, left then
                 * right child order with the root node at index zero.
                 *
                 * In this case if you have an index for any particular node
                 * ('id' here) then 2 * id + 1 is the index for the left
                 * child and 2 * id + 2 is the index for the right child...
                 */
                id = (gradient < node.t) ? 2 * id + 1 : 2 * id + 2;

                node = tree->nodes[id];
            }

            /* NB: node->label_pr_idx is a base-one index since index zero
             * is reserved to indicate that the node is not a leaf node
             */
            float* pr_table =
                &tree->label_pr_tables[(node.label_pr_idx - 1) * n_labels];
            for (int n = 0; n < n_labels; ++n)
            {
                out_pr_table[n] += pr_table[n];
            }
        }

        for (int n = 0; n < n_labels; ++n)
        {
            out_pr_table[n] /= (float)data->n_trees;
        }
    }

    if (data->n_threads > 1)
    {
        pthread_exit(NULL);
    }

    return NULL;
}

template<typename FloatT>
float*
infer_labels(struct gm_logger* log,
             RDTree** forest,
             int n_trees,
             FloatT* depth_image,
             int width, int height,
             float* out_labels,
             bool use_threads)
{
    int n_labels = (int)forest[0]->header.n_labels;
    size_t output_size = width * height * n_labels * sizeof(float);
    float* output_pr = out_labels ? out_labels : (float*)xmalloc(output_size);
    memset(output_pr, 0, output_size);

    void* (*infer_labels_callback)(void* userdata);

    if (forest[0]->header.sample_uv_offsets_nearest &&
        forest[0]->header.sample_uv_z_in_mm)
    {
        gm_assert(log,
                  (forest[0]->header.bg_depth > (((float)INT16_MAX / 1000.0f) - 0.001) &&
                   forest[0]->header.bg_depth < (((float)INT16_MAX / 1000.0f) + 0.001)),
                  "Expected tree requiring discrete mm unit sampling to have bg_depth = 32.7m, not %f",
                  forest[0]->header.bg_depth);

        infer_labels_callback = infer_labels_2018_06_discrete_mm<FloatT>;
    } else {
        gm_assert(log,
                  (forest[0]->header.sample_uv_offsets_nearest == false &&
                   forest[0]->header.sample_uv_z_in_mm == false),
                  "Unsupported decision tree sampling requirement");
        gm_assert(log,
                  forest[0]->header.bg_depth == 1000.0,
                  "Expected legacy decision tree to have bg_depth of 1000.0m");
        infer_labels_callback = infer_labels_2017_floor_uv_float_m<FloatT>;
    }

    int n_threads = std::thread::hardware_concurrency();
    if (!use_threads || n_threads <= 1)
    {
        InferThreadData data = {
            1, 1, forest, n_trees,
            (void*)depth_image, width, height, output_pr
        };
        infer_labels_callback((void*)(&data));
    }
    else
    {
        pthread_t threads[n_threads];
        InferThreadData data[n_threads];

        for (int i = 0; i < n_threads; ++i)
        {
            data[i] = { i, n_threads, forest, n_trees,
                (void*)depth_image, width, height, output_pr };
            if (pthread_create(&threads[i], NULL, infer_labels_callback,
                               (void*)(&data[i])) != 0)
            {
                gm_error(log,
                         "Error creating thread, results will be incomplete.\n");
                n_threads = i;
                break;
            }
        }

        for (int i = 0; i < n_threads; ++i)
        {
            if (pthread_join(threads[i], NULL) != 0)
                gm_error(log, "Error joining thread, trying to continue...\n");
        }
    }

    return output_pr;
}

template float*
infer_labels<half>(struct gm_logger* log,
                   RDTree**, int, half*, int, int, float*,
                   bool);
template float*
infer_labels<float>(struct gm_logger* log,
                    RDTree**, int, float*, int, int, float*,
                    bool);
