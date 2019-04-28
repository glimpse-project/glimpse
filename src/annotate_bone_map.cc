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

#include <getopt-compat.h>

#include <cmath>
#include <queue>
#include <utility>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include "xalloc.h"
#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_data.h"

typedef struct {
    int head;
    int tail;
    int parent;
} Bone;

typedef struct {
    FILE       *log_fp;
    struct gm_logger *log;

    bool        verbose;       // Verbose output
    int         n_threads;     // Number of threads to use for work

    int         n_joints;      // Number of joints
    int         n_bones;       // Number of bones
    int         n_sets;        // Number of joint position sets
    float*      joints;        // List of joint positions

    JSON_Value* bone_map;      // JSON nested bone map
    std::vector<Bone> bones;   // Description of each bone
    std::vector<glm::quat> avg_bone_rot; // Average bone rotations
} TrainContext;

typedef struct {
    glm::quat first_rotation;       // Bone rotation for the first dataset
    glm::vec4 cumulative_rotation;  // Cumulative rotation for all frames
    float radius;                   // Radius required from average rotation to
                                    // encompass all rotations
} RotationData;

typedef struct {
    TrainContext* ctx;
    int           start;         // Index to start analysing
    int           end;           // Index to end analysing
    float*        min_lengths;   // Minimum lengths of each bone
    float*        mean_lengths;  // Mean length of each bone
    float*        max_lengths;   // Maximum length of each bone
    RotationData  *rotation;     // Bone rotation data
    glm::quat*    first_rot;     // Bone rotations of the first frame
    glm::vec4*    acc_rot;       // Cumulative bone rotations
    int*          n_rots;        // Number of accumulated rotations
    float*        radius;        // Radius from the average rotation that
                                 // encompasses every rotation
} ThreadContext;

static glm::quat
calc_bone_rotation(TrainContext* ctx, Bone& bone, int joint_idx)
{
    Bone& parent = ctx->bones[bone.parent];
    float* joints = &ctx->joints[joint_idx];

    float* head = &joints[bone.head * 3];
    float* tail = &joints[bone.tail * 3];

    float* parent_head = &joints[parent.head * 3];
    float* parent_tail = &joints[parent.tail * 3];

    glm::vec3 bone_vec = glm::normalize(
        glm::vec3(tail[0] - head[0],
                  tail[1] - head[1],
                  tail[2] - head[2]));

    glm::vec3 parent_vec = glm::normalize(
        glm::vec3(parent_tail[0] - parent_head[0],
                  parent_tail[1] - parent_head[1],
                  parent_tail[2] - parent_head[2]));

    glm::vec3 axis = glm::normalize(glm::cross(bone_vec, parent_vec));
    float angle = acosf(glm::dot(bone_vec, parent_vec));
    while (angle > M_PI) angle -= 2 * M_PI;

    return glm::normalize(glm::angleAxis(angle, axis));
}

static void*
thread_stage1(void* userdata)
{
    ThreadContext* ctx = (ThreadContext*)userdata;

    for (int i = ctx->start; i < ctx->end; i++) {
        int joint_idx = i * ctx->ctx->n_joints * 3;
        float* joints = &ctx->ctx->joints[joint_idx];

        for (int j = 0; j < ctx->ctx->n_bones; ++j) {
            Bone& bone = ctx->ctx->bones[j];

            // Measure the length of the bone and update our running totals
            float* head = &joints[bone.head * 3];
            float* tail = &joints[bone.tail * 3];
            float dist = sqrtf(powf(tail[0] - head[0], 2.f) +
                               powf(tail[1] - head[1], 2.f) +
                               powf(tail[2] - head[2], 2.f));
            if (i == ctx->start) {
                ctx->min_lengths[j] = dist;
                ctx->max_lengths[j] = dist;
            } else {
                if (dist < ctx->min_lengths[j]) {
                    ctx->min_lengths[j] = dist;
                } else if (dist > ctx->max_lengths[j]) {
                    ctx->max_lengths[j] = dist;
                }
            }
            ctx->mean_lengths[j] += dist / ctx->ctx->n_sets;

            // Work out the angle between this bone and the parent bone and
            // alter the rotation range to include it
            if (j == 0) {
                continue;
            }

            glm::quat bone_rotation = calc_bone_rotation(ctx->ctx, bone, joint_idx);

            // Accumulate the bone rotations
            if (i == ctx->start) {
                ctx->rotation[j].first_rotation = bone_rotation;
            }

            if (glm::dot(ctx->rotation[j].first_rotation,
                         bone_rotation) < 0.f)
            {
                bone_rotation = -bone_rotation;
            }

            ctx->rotation[j].cumulative_rotation[0] += bone_rotation.w;
            ctx->rotation[j].cumulative_rotation[1] += bone_rotation.x;
            ctx->rotation[j].cumulative_rotation[2] += bone_rotation.y;
            ctx->rotation[j].cumulative_rotation[3] += bone_rotation.z;
        }
    }

    pthread_exit(NULL);
}

static void*
thread_stage2(void* userdata)
{
    ThreadContext* ctx = (ThreadContext*)userdata;

    for (int i = ctx->start; i < ctx->end; i++) {
        int joint_idx = i * ctx->ctx->n_joints * 3;

        for (int j = 1; j < ctx->ctx->n_bones; ++j) {
            Bone& bone = ctx->ctx->bones[j];

            glm::quat bone_rotation = calc_bone_rotation(ctx->ctx, bone, joint_idx);
            float diff = glm::angle(bone_rotation *
                                    glm::inverse(ctx->ctx->avg_bone_rot[j-1]));
            while (diff > M_PI) diff -= 2 * M_PI;
            diff = fabsf(diff);

            if (diff > ctx->rotation[j].radius) {
                ctx->rotation[j].radius = diff;
            }
        }
    }

    pthread_exit(NULL);
}

static void
print_usage(FILE* stream)
{
    fprintf(stream,
            "Usage: annotate_bone_map [OPTIONS] <data dir> <index name> <joint map> <bone map> [out_file]\n"
            "\n"
            "Determine the min, mean and max lengths of each bone over a set of\n"
            "motion capture data and output a JSON file with the data.\n"
            "If the output file is omitted, the bone map will be overwritten.\n"
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
    const char* bone_map_path;
    const char* out_file;

    bool pretty = false;

    TrainContext ctx = {};
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
    bone_map_path = argv[optind + 3];
    out_file = ((argc - optind) > 4) ? argv[optind + 4] : bone_map_path;

    // Load bone map
    ctx.bone_map = json_parse_file(bone_map_path);
    if (!ctx.bone_map || json_value_get_type(ctx.bone_map) != JSONObject) {
        fprintf(stderr, "Error parsing bone map\n");
        return 1;
    }

    // Load in joint position data
    printf("Scanning training directories...\n");
    JSON_Value *meta =
        gm_data_load_simple(ctx.log,
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

    // Load joint map. Do it after loading data to avoid needing to do
    // validation. I guess it would be nice if gm_data could return it and we
    // could avoid parsing it twice, but... I'm lazy?
    JSON_Value* joint_map = json_parse_file(joint_map_path);
    JSON_Array* joint_array = json_array(joint_map);

    // Validate and enumerate bones
    std::queue<std::pair<JSON_Object*, int>> bones;
    bones.push({json_object(ctx.bone_map), -1});
    while (!bones.empty()) {
        std::pair<JSON_Object*, int>& bone_and_parent = bones.front();
        JSON_Object* bone = bone_and_parent.first;
        int parent = bone_and_parent.second;
        bones.pop();

        // Validate bone
        if (!json_object_has_value(bone, "head") ||
            !json_object_has_value(bone, "tail"))
        {
            fprintf(stderr, "Bone missing required properties\n");
            return 1;
        }

        // Find joint indices
        const char* head_name = json_object_get_string(bone, "head");
        const char* tail_name = json_object_get_string(bone, "tail");

        int head = -1;
        int tail = -1;
        for (int i = 0; i < json_array_get_count(joint_array) &&
             (head < 0 || tail < 0); ++i)
        {
            JSON_Object* joint = json_array_get_object(joint_array, i);
            const char* joint_name = json_object_get_string(joint, "joint");
            if (head < 0 && strcmp(joint_name, head_name) == 0) {
                head = i;
                continue;
            }
            if (tail < 0 && strcmp(joint_name, tail_name) == 0) {
                tail = i;
                continue;
            }
        }
        if (head == -1 || tail == -1) {
            fprintf(stderr, "Bone '%s'->'%s' joints not found",
                    head_name, tail_name);
            return 1;
        }

        ctx.bones.push_back({head, tail, parent});

        // Collect next bones
        JSON_Array* children = json_object_get_array(bone, "children");
        if (children) {
            for (int i = 0; i < json_array_get_count(children); ++i) {
                bones.push({json_array_get_object(children, i), ctx.n_bones});
            }
        }

        // Increment bone count
        ++ctx.n_bones;
    }

    // Free joint map we don't need anymore
    json_value_free(joint_map);

    // Create worker threads
    pthread_t threads[ctx.n_threads];
    ThreadContext thread_ctx[ctx.n_threads];

    float sets_per_thread = ctx.n_sets / (float)ctx.n_threads;
    float error = 0.f;
    int start = 0;

    for (int i = 0; i < ctx.n_threads; i++) {
        thread_ctx[i].ctx = &ctx;

        thread_ctx[i].start = start;
        if (i < ctx.n_threads - 1) {
            thread_ctx[i].end = (int)(start + sets_per_thread + error);
            error += sets_per_thread - (thread_ctx[i].end - thread_ctx[i].start);
            start = thread_ctx[i].end;
        } else {
            thread_ctx[i].end = ctx.n_sets;
        }

        thread_ctx[i].min_lengths = (float*)xcalloc(ctx.n_bones, sizeof(float));
        thread_ctx[i].mean_lengths = (float*)xcalloc(ctx.n_bones, sizeof(float));
        thread_ctx[i].max_lengths = (float*)xcalloc(ctx.n_bones, sizeof(float));
        thread_ctx[i].rotation = (RotationData*)xcalloc(ctx.n_bones, sizeof(RotationData));

        if (pthread_create(&threads[i], NULL, thread_stage1,
                           (void*)(&thread_ctx[i])) != 0)
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    // Wait for threads to finish and collate the data
    for (int i = 0; i < ctx.n_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread, trying to continue...\n");
        }

        if (i == 0) {
            continue;
        }

        for (int j = 0; j < ctx.n_bones; ++j) {
            if (thread_ctx[i].min_lengths[j] < thread_ctx[0].min_lengths[j]) {
                thread_ctx[0].min_lengths[j] = thread_ctx[i].min_lengths[j];
            }
            if (thread_ctx[i].max_lengths[j] > thread_ctx[0].max_lengths[j]) {
                thread_ctx[0].max_lengths[j] = thread_ctx[i].max_lengths[j];
            }
            thread_ctx[0].mean_lengths[j] += thread_ctx[i].mean_lengths[j];

            if (j != 0) {
                thread_ctx[0].rotation[j].cumulative_rotation +=
                    thread_ctx[i].rotation[j].cumulative_rotation;
            }
        }
    }

    // Calculate the average bone rotations
    float quat_scale = 1.f / ctx.n_sets;
    for (int i = 1; i < ctx.n_bones; ++i) {
        ctx.avg_bone_rot.push_back(glm::normalize(
            glm::quat(thread_ctx[0].rotation[i].cumulative_rotation[0] * quat_scale,
                      thread_ctx[0].rotation[i].cumulative_rotation[1] * quat_scale,
                      thread_ctx[0].rotation[i].cumulative_rotation[2] * quat_scale,
                      thread_ctx[0].rotation[i].cumulative_rotation[3] * quat_scale)));
    }

    // Run the second stage
    for (int i = 0; i < ctx.n_threads; i++) {
        if (pthread_create(&threads[i], NULL, thread_stage2,
                           (void*)(&thread_ctx[i])) != 0)
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    for (int i = 0; i < ctx.n_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread, trying to continue...\n");
        }

        if (i == 0) {
            continue;
        }

        // Determine the max radius
        for (int j = 1; j < ctx.n_bones; ++j) {
            if (thread_ctx[i].rotation[j].radius >
                thread_ctx[0].rotation[j].radius)
            {
                thread_ctx[0].rotation[j].radius =
                    thread_ctx[i].rotation[j].radius;
            }
        }
    }

    // Output to file
    JSON_Value* root = json_value_init_object();
    std::queue<JSON_Object*> bones_annotated;
    bones_annotated.push(json_object(root));

    std::queue<JSON_Object*> bones_orig;
    bones_orig.push(json_object(ctx.bone_map));

    int bone_id = 0;
    while (!bones_orig.empty()) {
        JSON_Object* bone = bones_orig.front();
        bones_orig.pop();
        JSON_Object* bone_annotated = bones_annotated.front();
        bones_annotated.pop();

        // Copy head/tail names
        const char* head_name = json_object_get_string(bone, "head");
        const char* tail_name = json_object_get_string(bone, "tail");
        json_object_set_string(bone_annotated, "head", head_name);
        json_object_set_string(bone_annotated, "tail", tail_name);

        // Add calculated bone length data
        json_object_set_number(bone_annotated, "min_length",
                               (double)thread_ctx[0].min_lengths[bone_id]);
        json_object_set_number(bone_annotated, "mean_length",
                               (double)thread_ctx[0].mean_lengths[bone_id]);
        json_object_set_number(bone_annotated, "max_length",
                               (double)thread_ctx[0].max_lengths[bone_id]);

        // Add calculated bone rotation restriction data
        if (bone_id != 0) {
            JSON_Value* constraint = json_value_init_object();
            JSON_Value* rotation = json_value_init_object();
            json_object_set_number(json_object(rotation), "w",
                                   ctx.avg_bone_rot[bone_id-1].w);
            json_object_set_number(json_object(rotation), "x",
                                   ctx.avg_bone_rot[bone_id-1].x);
            json_object_set_number(json_object(rotation), "y",
                                   ctx.avg_bone_rot[bone_id-1].y);
            json_object_set_number(json_object(rotation), "z",
                                   ctx.avg_bone_rot[bone_id-1].z);
            json_object_set_value(json_object(constraint), "rotation", rotation);
            json_object_set_number(json_object(constraint), "radius",
                                   thread_ctx[0].rotation[bone_id].radius);

            json_object_set_value(bone_annotated, "rotation_constraint",
                                  constraint);
        }

        if (ctx.verbose)
        {
            printf("    Bone %s->%s - min:  %.2f\n"
                   "                  mean: %.2f\n"
                   "                  max:  %.2f\n"
                   "                  rotation constraint: (x,y,z,w - radius)\n"
                   "                  %.2f, %.2f, %.2f, %.2f - %.2f\n",
                   head_name, tail_name,
                   thread_ctx[0].min_lengths[bone_id],
                   thread_ctx[0].mean_lengths[bone_id],
                   thread_ctx[0].max_lengths[bone_id],
                   ctx.avg_bone_rot[bone_id-1].x,
                   ctx.avg_bone_rot[bone_id-1].y,
                   ctx.avg_bone_rot[bone_id-1].z,
                   ctx.avg_bone_rot[bone_id-1].w,
                   glm::degrees(thread_ctx[0].rotation[bone_id].radius));
        }

        // Prime child objects
        JSON_Array* children = json_object_get_array(bone, "children");
        if (children) {
            JSON_Value* children_annotated = json_value_init_array();
            json_object_set_value(bone_annotated, "children",
                                  children_annotated);

            for (int i = 0; i < json_array_get_count(children); ++i) {
                bones_orig.push(json_array_get_object(children, i));

                JSON_Value* child = json_value_init_object();
                json_array_append_value(json_array(children_annotated), child);
                bones_annotated.push(json_object(child));
            }
        }

        // Increment bone index
        ++bone_id;
    }

    // Free bone map we no longer need
    json_value_free(ctx.bone_map);

    // Serialize json to file
    if (pretty) {
        json_serialize_to_file_pretty(root, out_file);
    } else {
        json_serialize_to_file(root, out_file);
    }

    // Free data
    json_value_free(root);
    for (int i = 0; i < ctx.n_threads; i++)
    {
        xfree(thread_ctx[i].min_lengths);
        xfree(thread_ctx[i].mean_lengths);
        xfree(thread_ctx[i].max_lengths);
        xfree(thread_ctx[i].rotation);
    }
    xfree(ctx.joints);

    json_value_free(meta);

    return 0;
}
