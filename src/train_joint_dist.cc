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
#include <getopt.h>

#include <cmath>

#include "xalloc.h"
#include "train_utils.h"
#include "parson.h"

#include "glimpse_log.h"

typedef struct {
    FILE       *log_fp;
    struct gm_logger *log;

    bool        verbose;       // Verbose output
    int         n_threads;     // Number of threads to use for work

    int         n_joints;      // Number of joints
    int         n_sets;        // Number of joint position sets
    float*      joints;        // List of joint positions
} TrainContext;

typedef struct {
    TrainContext* ctx;
    int           start;       // Index to start analysing
    int           end;         // Index to end analysing
    float*        min_dists;   // Minimum distance between each pair of joints
    float*        mean_dists;  // Mean distance between each pair of joints
    float*        max_dists;   // Maximum distance between each pair of joints
} ThreadContext;

static void*
thread_body(void* userdata)
{
    ThreadContext* ctx = (ThreadContext*)userdata;

    for (int i = ctx->start; i < ctx->end; i++)
    {
        int joint_idx = i * ctx->ctx->n_joints * 3;
        float* joints = &ctx->ctx->joints[joint_idx];

        for (int j = 0; j < ctx->ctx->n_joints; j++)
        {
            float* joint1 = &joints[j * 3];
            for (int k = 0; k < ctx->ctx->n_joints; k++)
            {
                int result_idx = j * ctx->ctx->n_joints + k;
                float* joint2 = &joints[k * 3];
                float dist = sqrtf(powf(joint2[0] - joint1[0], 2.f) +
                                   powf(joint2[1] - joint1[1], 2.f) +
                                   powf(joint2[2] - joint1[2], 2.f));
                if (i == ctx->start)
                {
                    ctx->min_dists[result_idx] = dist;
                    ctx->max_dists[result_idx] = dist;
                }
                else
                {
                    if (dist < ctx->min_dists[result_idx])
                    {
                        ctx->min_dists[result_idx] = dist;
                    }
                    else if (dist > ctx->max_dists[result_idx])
                    {
                        ctx->max_dists[result_idx] = dist;
                    }
                }
                ctx->mean_dists[result_idx] += dist / ctx->ctx->n_sets;
            }
        }
    }

    pthread_exit(NULL);
}

static void
print_usage(FILE* stream)
{
    fprintf(stream,
            "Usage: train_joint_dist [OPTIONS] <data dir> <index name> <joint map> <out_file>\n"
            "\n"
            "Determine the min, mean and max distances between each joint over a set of\n"
            "motion capture data and output a JSON file with the data.\n"
            "\n"
            "  -j, --threads=NUMBER        Number of threads to use (default: autodetect)\n"
            "  -p, --pretty                Output prettified JSON\n"
            "  -v, --verbose               Verbose output\n"
            "  -h, --help                  Display this message\n");
}

int
main(int argc, char** argv)
{
    int opt;

    const char* data_dir;
    const char* index_name;
    const char* joint_map_path;
    const char* out_file;

    bool pretty = false;

    TrainContext ctx = { 0, };
    ctx.n_threads = std::thread::hardware_concurrency();

    ctx.log = gm_logger_new(NULL, NULL);

    const char *short_opts="+jpvh";
    const struct option long_opts[] = {
        {"threads",         required_argument,  0, 'j'},
        {"pretty",          no_argument,        0, 'p'},
        {"verbose",         no_argument,        0, 'v'},
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1)
    {
        switch (opt)
        {
        case 'j':
            ctx.n_threads = atoi(optarg);
            break;
        case 'p':
            pretty = true;
            break;
        case 'v':
            ctx.verbose = true;
            break;
        case 'h':
            print_usage(stdout);
            return 0;
        }
    }

    if ((argc - optind) < 4) {
        print_usage(stderr);
        return 1;
    }

    data_dir = argv[optind];
    index_name = argv[optind + 1];
    joint_map_path = argv[optind + 2];
    out_file = argv[optind + 3];

    printf("Scanning training directories...\n");
    JSON_Value *meta =
        gather_train_data(ctx.log,
                          data_dir,
                          index_name,
                          joint_map_path,
                          &ctx.n_sets,
                          &ctx.n_joints,
                          NULL, NULL, // width, height
                          NULL, // depth images
                          NULL, // label images
                          &ctx.joints,
                          NULL); // simply abort on error

    // Create worker threads
    pthread_t threads[ctx.n_threads];
    ThreadContext thread_ctx[ctx.n_threads];

    float sets_per_thread = ctx.n_sets / (float)ctx.n_threads;
    float error = 0.f;
    int start = 0;
    size_t array_size = ctx.n_joints * ctx.n_joints;

    for (int i = 0; i < ctx.n_threads; i++) {
        thread_ctx[i].ctx = &ctx;

        thread_ctx[i].start = start;
        if (i < ctx.n_threads - 1)
        {
            thread_ctx[i].end = (int)(start + sets_per_thread + error);
            error += sets_per_thread - (thread_ctx[i].end - thread_ctx[i].start);
            start = thread_ctx[i].end;
        }
        else
        {
            thread_ctx[i].end = ctx.n_sets;
        }

        thread_ctx[i].min_dists = (float*)xcalloc(array_size, sizeof(float));
        thread_ctx[i].mean_dists = (float*)xcalloc(array_size, sizeof(float));
        thread_ctx[i].max_dists = (float*)xcalloc(array_size, sizeof(float));

        if (pthread_create(&threads[i], NULL, thread_body,
                           (void*)(&thread_ctx[i])) != 0)
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    // Wait for threads to finish and collate the data
    for (int i = 0; i < ctx.n_threads; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            fprintf(stderr, "Error joining thread, trying to continue...\n");
        }

        if (i == 0) {
            continue;
        }

        for (int j = 0; j < ctx.n_joints; j++)
        {
            for (int k = 0; k < ctx.n_joints; k++)
            {
                int result_idx = j * ctx.n_joints + k;
                if (thread_ctx[i].min_dists[result_idx] <
                    thread_ctx[0].min_dists[result_idx])
                {
                    thread_ctx[0].min_dists[result_idx] =
                        thread_ctx[i].min_dists[result_idx];
                }
                if (thread_ctx[i].max_dists[result_idx] >
                    thread_ctx[0].max_dists[result_idx])
                {
                    thread_ctx[0].max_dists[result_idx] =
                        thread_ctx[i].max_dists[result_idx];
                }
                thread_ctx[0].mean_dists[result_idx] +=
                    thread_ctx[i].mean_dists[result_idx];
            }
        }
    }

    // Output to file
    JSON_Value* root = json_value_init_array();

    for (int j = 0; j < ctx.n_joints; j++)
    {
        JSON_Value* joint_array = json_value_init_array();

        if (ctx.verbose)
        {
            printf("Joint %01d\n"
                   "-------\n", (int)j);
        }

        for (int k = 0; k < ctx.n_joints; k++)
        {
            int idx = j * ctx.n_joints + k;

            if (ctx.verbose)
            {
                printf("    Joint %01d - min:  %.2f\n"
                       "              mean: %.2f\n"
                       "              max:  %.2f\n",
                       (int)k,
                       thread_ctx[0].min_dists[idx],
                       thread_ctx[0].mean_dists[idx],
                       thread_ctx[0].max_dists[idx]);
            }

            JSON_Value* object = json_value_init_object();
            json_object_set_number(json_object(object), "min",
                                   (double)thread_ctx[0].min_dists[idx]);
            json_object_set_number(json_object(object), "mean",
                                   (double)thread_ctx[0].mean_dists[idx]);
            json_object_set_number(json_object(object), "max",
                                   (double)thread_ctx[0].max_dists[idx]);

            json_array_append_value(json_array(joint_array), object);
        }

        if (ctx.verbose) {
            printf("\n");
        }

        json_array_append_value(json_array(root), joint_array);
    }

    if (pretty) {
        json_serialize_to_file_pretty(root, out_file);
    } else {
        json_serialize_to_file(root, out_file);
    }

    // Free data
    json_value_free(root);
    for (int i = 0; i < ctx.n_threads; i++)
    {
        xfree(thread_ctx[i].min_dists);
        xfree(thread_ctx[i].mean_dists);
        xfree(thread_ctx[i].max_dists);
    }
    xfree(ctx.joints);

    json_value_free(meta);

    return 0;
}
