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


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <vector>

#include "half.hpp"
#include "train_utils.h"
#include "image_utils.h"
#include "xalloc.h"
#include "llist.h"
#include "parson.h"

#include "glimpse_log.h"

using half_float::half;

#define xsnprintf(dest, fmt, ...) do { \
        if (snprintf(dest, sizeof(dest), fmt,  __VA_ARGS__) >= (int)sizeof(dest)) \
            exit(1); \
    } while(0)

typedef struct {
    char *labels_path;
    char *depth_path;
    char *joints_path;
} FramePaths;

typedef struct {
    struct gm_logger *log;
    double   vertical_fov;  // Field of view used to render depth images
    int      n_images;      // Number of training images
    int      n_joints;      // Number of joints
    std::vector<FramePaths> paths; // Array of label, depth and joint file paths
    IUImageSpec label_spec; // Label image specification
    IUImageSpec depth_spec; // Depth image specification
    half*    depth_images;  // Depth image data
    uint8_t* label_images;  // Label image data
    float*   joint_data;    // Joint data
    bool     gather_depth;  // Whether to load depth images
    bool     gather_label;  // Whether to load label images
    bool     gather_joints; // Whether to gather joint data
} TrainData;

static bool
load_training_index(const char* top_src_dir,
                    const char* index_name,
                    TrainData* data,
                    char **err)
{
    char index_filename[1024];
    bool cont = true;

    xsnprintf(index_filename, "%s/index.%s", top_src_dir, index_name);

    FILE* fp = fopen(index_filename, "r");
    if (!fp) {
        gm_throw(data->log, err, "Failed to open index %s\n", index_filename);
        return false;
    }

    char* line = NULL;
    size_t line_buf_len = 0;
    int line_len;
    while (cont && (line_len = getline(&line, &line_buf_len, fp)) != -1)
    {
        if (line_len <= 1)
            continue;

        /* remove the trailing newline from the line */
        line[line_len - 1] = '\0';

        FramePaths paths;
        xasprintf(&paths.labels_path, "%s/labels/%s.png", top_src_dir, line);
        xasprintf(&paths.depth_path, "%s/depth/%s.exr", top_src_dir, line);
        xasprintf(&paths.joints_path, "%s/labels/%s.jnt", top_src_dir, line);
        data->paths.push_back(paths);
        data->n_images++;
    }

    free(line);

    fclose(fp);

    return true;
}

static bool
load_frame(TrainData *data, int index, char **err)
{
    char* label_path = data->paths[index].labels_path;
    char* depth_path = data->paths[index].depth_path;
    char* joint_path = data->paths[index].joints_path;

    // Read label image
    if (label_path && data->gather_label)
    {
        uint8_t* output = &data->label_images[
            index * data->label_spec.width * data->label_spec.height];
        if (iu_read_png_from_file(label_path, &data->label_spec, &output,
                                  NULL, // palette output
                                  NULL) // palette size
            != SUCCESS)
        {
            gm_throw(data->log, err, "Failed to read image '%s'\n", label_path);
            return false;
        }
    }

    // Read depth image
    if (depth_path && data->gather_depth)
    {
        void* output = &data->depth_images[
            index * data->depth_spec.width * data->depth_spec.height];
        if (iu_read_exr_from_file(depth_path, &data->depth_spec, &output) !=
            SUCCESS)
        {
            gm_throw(data->log, err, "Failed to read image '%s'\n", depth_path);
            return false;
        }
    }

    // Read joint data
    if (joint_path && data->gather_joints)
    {
        FILE* fp;
        if (!(fp = fopen(joint_path, "rb")))
        {
            gm_throw(data->log, err, "Error opening joint file '%s'\n", joint_path);
            return false;
        }

        if (fseek(fp, 0, SEEK_END) == -1)
        {
            gm_throw(data->log, err, "Error seeking to end of joint file '%s'\n",
                    joint_path);
            return false;
        }

        long n_bytes = ftell(fp);
        if (n_bytes == 0 ||
            n_bytes % sizeof(float) != 0 ||
            (n_bytes % sizeof(float)) % 3 != 0)
        {
            gm_throw(data->log, err, "Unexpected joint file size in '%s'\n",
                    joint_path);
            return false;
        }

        uint8_t n_joints = (uint8_t)((n_bytes / sizeof(float)) / 3);
        if (n_joints != data->n_joints)
        {
            gm_throw(data->log, err,
                     "Unexpected number of joints %u (expected %u)\n",
                     n_joints, data->n_joints);
            return false;
        }

        if (fseek(fp, 0, SEEK_SET) == -1)
        {
            gm_throw(data->log, err,
                     "Error seeking to start of joint file '%s'\n", joint_path);
            return false;
        }

        float* joints = &data->joint_data[index * n_joints * 3];
        if (fread(joints, sizeof(float) * 3, n_joints, fp) != n_joints)
        {
            gm_throw(data->log, err, "%s: Error reading joints\n", joint_path);
            return false;
        }

        if (fclose(fp) != 0)
        {
            gm_throw(data->log, err, "Error closing joint file '%s'\n", joint_path);
            return false;
        }
    }

    return true;
}

bool
gather_train_data(struct gm_logger* log,
                  const char* data_dir,
                  const char* index_name,
                  const char* joint_map_path,
                  int* out_n_images,
                  int* out_n_joints,
                  int* out_width,
                  int* out_height,
                  half** out_depth_images,
                  uint8_t** out_label_images,
                  float** out_joints,
                  int* out_n_labels,
                  float* out_fov,
                  char** err)
{
    char meta_filename[1024];

    TrainData data = {
        log,
        0,                                    // Field of view used to render depth
        0,                                    // Number of training images
        0,                                    // Number of joints
        {},                                   // Image paths vector
        {0,0,IU_FORMAT_U8},                   // Label image specification
        {0,0,IU_FORMAT_HALF},                 // Depth image specification
        NULL,                                 // Depth image data
        NULL,                                 // Label image data
        NULL,                                 // Joint data
        !!out_depth_images,                   // Whether to gather depth images
        !!out_label_images,                   // Whether to gather label images
        (joint_map_path && out_joints)        // Whether to gather joint data
    };

    xsnprintf(meta_filename, "%s/meta.json", data_dir);

    JSON_Value* meta = json_parse_file(meta_filename);
    if (!meta) {
        gm_throw(log, err, "Failed to parse %s", meta_filename);
        return false;
    }

    int n_labels = json_object_get_number(json_object(meta), "n_labels");
    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");

    data.vertical_fov = json_object_get_number(camera, "vertical_fov");
    int width = json_object_get_number(camera, "width");
    int height = json_object_get_number(camera, "height");

    json_value_free(meta);
    meta = NULL;

    data.label_spec.width = width;
    data.label_spec.height = height;
    data.depth_spec.width = width;
    data.depth_spec.height = height;

    if (data.gather_joints)
    {
        JSON_Value *map = json_parse_file(joint_map_path);
        if (!map) {
            gm_throw(log, err, "Failed to parse joint map %s\n", joint_map_path);
            return false;
        }

        /* For now we just care about how many joints there are but maybe
         * we should be handing the map back to the caller somehow?
         */
        data.n_joints = json_array_get_count(json_array(map));
        json_value_free(map);
        map = NULL;
    }

    if (!load_training_index(data_dir, index_name, &data, err))
        return false;

    size_t n_pixels = width * height * data.n_images;

    if (data.gather_label)
        data.label_images = (uint8_t*)xmalloc(n_pixels * sizeof(uint8_t));

    if (data.gather_depth)
        data.depth_images = (half*)xmalloc(n_pixels * sizeof(half));

    if (data.gather_joints) {
        data.joint_data = (float*)xmalloc(data.n_images * data.n_joints *
                                          3 * sizeof(float));
    }

    *out_n_images = data.n_images;
    gm_info(log, "Processing %d training images...\n", *out_n_images);

    bool load_ok = true;
    for (int i = 0; load_ok == true && i < data.paths.size(); i++) {
        load_ok = load_frame(&data, i, err);
    }
    for (int i = 0; i < data.paths.size(); i++) {
        xfree(data.paths[i].labels_path);
        xfree(data.paths[i].depth_path);
        xfree(data.paths[i].joints_path);
    }
    data.paths.resize(0);
    if (!load_ok) {
        xfree(data.label_images);
        xfree(data.depth_images);
        xfree(data.joint_data);
        return false;
    }

    if (out_width) {
        *out_width = data.gather_label ?
            data.label_spec.width : data.depth_spec.width;
    }
    if (out_height) {
        *out_height = data.gather_label ?
            data.label_spec.height : data.depth_spec.height;
    }
    if (out_n_joints) {
        *out_n_joints = data.n_joints;
    }
    if (out_depth_images) {
        *out_depth_images = data.depth_images;
    }
    if (out_label_images) {
        *out_label_images = data.label_images;
    }
    if (out_joints) {
        *out_joints = data.joint_data;
    }
    if (out_n_labels) {
        *out_n_labels = n_labels;
    }
    if (out_fov) {
        *out_fov = data.vertical_fov;
    }

    return true;
}

