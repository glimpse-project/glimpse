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
#include <string.h>

#include "infer_labels.h"
#include "xalloc.h"
#include "rdt_tree.h"


typedef struct {
    int thread;
    int n_threads;
    RDTree** forest;
    int n_trees;
    void* depth_image;
    int width;
    int height;
    float* output;
    bool flip;
} InferThreadData;

typedef vector(int, 2) Int2D;

static void*
infer_label_probs_cb(void* userdata)
{
    InferThreadData* data = (InferThreadData*)userdata;
    float* depth_image = (float*)data->depth_image;
    int n_labels = data->forest[0]->header.n_labels;

    float bg_depth = data->forest[0]->header.bg_depth;
    int bg_label = data->forest[0]->header.bg_label;

    int width = data->width;
    int height = data->height;

    // Accumulate probability map
    for (int off = data->thread;
         off < width * height;
         off += data->n_threads)
    {
        int y = off / data->width;
        int x = off % data->width;

        int out_pr_idx = off * n_labels;
        float depth = depth_image[off];

        if (depth >= bg_depth) {
            (data->output + out_pr_idx)[bg_label] += 1.f;
            continue;
        }

        Int2D pixel = { x, y };
        for (int i = 0; i < data->n_trees; ++i)
        {
            RDTree* tree = data->forest[i];
            uint8_t* flip_map = tree->header.flip_map;

            for (int j = 0; j < (data->flip ? 2 : 1); ++j) {
                int id = 0;
                Node node = tree->nodes[0];
                bool flip = (j == 1);

                while (node.label_pr_idx == 0) {
                    Int2D u, v;
                    if (flip) {
                        u = (Int2D){ (int)(pixel[0] - node.uv[0] / depth),
                                     (int)(pixel[1] + node.uv[1] / depth) };
                        v = (Int2D){ (int)(pixel[0] - node.uv[2] / depth),
                                     (int)(pixel[1] + node.uv[3] / depth) };
                    } else {
                        u = (Int2D){ (int)(pixel[0] + node.uv[0] / depth),
                                     (int)(pixel[1] + node.uv[1] / depth) };
                        v = (Int2D){ (int)(pixel[0] + node.uv[2] / depth),
                                     (int)(pixel[1] + node.uv[3] / depth) };
                    }

                    float upixel = (u[0] >= 0 && u[0] < (int)width &&
                                    u[1] >= 0 && u[1] < (int)height) ?
                        (float)depth_image[((u[1] * width) + u[0])] : bg_depth;
                    float vpixel = (v[0] >= 0 && v[0] < (int)width &&
                                    v[1] >= 0 && v[1] < (int)height) ?
                        (float)depth_image[((v[1] * width) + v[0])] : bg_depth;

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
                float* out_pr_table = &data->output[out_pr_idx];
                if (flip) {
                    for (int n = 0; n < n_labels; ++n) {
                        out_pr_table[flip_map[n]] += pr_table[n];
                    }
                } else {
                    for (int n = 0; n < n_labels; ++n) {
                        out_pr_table[n] += pr_table[n];
                    }
                }
            }
        }

        float divider = (float)
            (data->flip ? data->n_trees * 2 : data->n_trees);
        for (int n = 0; n < n_labels; ++n) {
            (data->output + out_pr_idx)[n] /= divider;
        }
    }

    if (data->n_threads > 1)
    {
        pthread_exit(NULL);
    }

    return NULL;
}

float*
infer_labels(struct gm_logger* log,
             RDTree** forest,
             int n_trees,
             float* depth_image,
             int width, int height,
             float* out_labels,
             bool use_threads,
             bool do_flip)
{
    int n_labels = (int)forest[0]->header.n_labels;
    size_t output_size = width * height * n_labels * sizeof(float);
    float* output_pr = out_labels;

    gm_assert(log, output_pr != NULL, "NULL output buffer for label probabilities");

    memset(output_pr, 0, output_size);

    void* (*infer_labels_callback)(void* userdata);
    infer_labels_callback = infer_label_probs_cb;

    int n_threads = std::thread::hardware_concurrency();
    if (!use_threads || n_threads <= 1)
    {
        InferThreadData data = {
            1, 1, forest, n_trees,
            (void*)depth_image, width, height, output_pr, do_flip
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
                (void*)depth_image, width, height, output_pr, do_flip };
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
