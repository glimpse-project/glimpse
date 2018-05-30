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
#include "parson.h"

#include "glimpse_log.h"

using half_float::half;

#define xsnprintf(dest, size, fmt, ...) do { \
        if (snprintf(dest, size, fmt,  __VA_ARGS__) >= (int)size) \
            exit(1); \
    } while(0)

typedef struct {
    struct gm_logger *log;

    int      n_images;      // Number of training images
    int      n_labels;
    int      n_joints;      // Number of joints

    int      width;
    int      height;
    double   vertical_fov;  // Field of view used to render depth images

    half*    depth_images;  // Depth image data
    uint8_t* label_images;  // Label image data
    float*   joint_data;    // Joint data

    bool     gather_depth;  // Whether to load depth images
    bool     gather_label;  // Whether to load label images
} TrainData;

struct gm_data_index
{
    struct gm_logger* log;
    JSON_Value *meta;
    char* top_dir;
    std::vector<char*> paths; // Array of frame paths
};

void
gm_data_index_destroy(struct gm_data_index* data_index)
{
    for (int i = 0; i < (int)data_index->paths.size(); i++)
        free(data_index->paths[i]);
    free(data_index->top_dir);
    if (data_index->meta)
        json_value_free(data_index->meta);
    delete data_index;
}

const char*
gm_data_index_get_top_dir(struct gm_data_index* data_index)
{
    return data_index->top_dir;
}

struct gm_data_index*
gm_data_index_open(struct gm_logger* log,
                   const char* top_dir,
                   const char* index_name,
                   char **err)
{
    struct gm_data_index* data_index = new gm_data_index();
    char index_filename[1024];
    bool cont = true;

    data_index->log = log;

    data_index->top_dir = strdup(top_dir);
    xsnprintf(index_filename, sizeof(index_filename), "%s/index.%s", top_dir, index_name);

    FILE* fp = fopen(index_filename, "r");
    if (!fp) {
        gm_throw(log, err, "Failed to open index %s\n", index_filename);
        gm_data_index_destroy(data_index);
        return NULL;
    }

    char meta_filename[1024];
    xsnprintf(meta_filename, sizeof(meta_filename), "%s/meta.json", top_dir);
    data_index->meta = json_parse_file(meta_filename);
    if (!data_index->meta) {
        gm_throw(log, err, "Failed to parse %s", meta_filename);
        gm_data_index_destroy(data_index);
        return NULL;
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

        data_index->paths.push_back(strdup(line));
    }

    free(line);

    fclose(fp);

    return data_index;
}

int
gm_data_index_get_len(struct gm_data_index* data_index)
{
    return (int)data_index->paths.size();
}

JSON_Value*
gm_data_index_get_meta(struct gm_data_index* data_index)
{
    return data_index->meta;
}

int
gm_data_index_get_width(struct gm_data_index* data_index)
{
    JSON_Value* meta = data_index->meta;
    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");
    return json_object_get_number(camera, "width");
}

int
gm_data_index_get_height(struct gm_data_index* data_index)
{
    JSON_Value* meta = data_index->meta;
    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");
    return json_object_get_number(camera, "height");
}

int
gm_data_index_get_n_labels(struct gm_data_index* data_index)
{
    return json_object_get_number(json_object(data_index->meta), "n_labels");
}

float
gm_data_index_get_vfov(struct gm_data_index* data_index)
{
    JSON_Value* meta = data_index->meta;
    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");
    return json_object_get_number(camera, "vertical_fov");
}

bool
gm_data_index_foreach(struct gm_data_index* data_index,
                      bool (*callback)(struct gm_data_index* data_index,
                                       int index,
                                       const char* frame_path,
                                       void* user_data,
                                       char** err),
                      void* user_data,
                      char** err)
{
    for (int i = 0; i < (int)data_index->paths.size(); i++) {
        if (!callback(data_index, i, data_index->paths[i], user_data, err))
            return false;
    }
    return true;
}

static bool
load_frame_foreach_cb(struct gm_data_index* data_index,
                      int index,
                      const char* path,
                      void* user_data,
                      char** err)
{
    TrainData* data = (TrainData*)user_data;
    const char* top_dir = gm_data_index_get_top_dir(data_index);

    char labels_filename[512];
    char depth_filename[512];

    xsnprintf(labels_filename, sizeof(labels_filename), "%s/labels/%s.png", top_dir, path);
    xsnprintf(depth_filename, sizeof(depth_filename), "%s/depth/%s.exr", top_dir, path);

    if (data->gather_label)
    {
        IUImageSpec label_spec = { data->width, data->height, IU_FORMAT_U8 };
        int64_t off = (int64_t)index * data->width * data->height;
        uint8_t* output = &data->label_images[off];

        if (iu_read_png_from_file(labels_filename, &label_spec, &output,
                                  NULL, // palette output
                                  NULL) // palette size
            != SUCCESS)
        {
            gm_throw(data->log, err, "Failed to read image '%s'\n", labels_filename);
            return false;
        }
    }

    if (data->gather_depth)
    {
        IUImageSpec depth_spec = { data->width, data->height, IU_FORMAT_HALF };
        int64_t off = (int64_t)index * data->width * data->height;
        void* output = &data->depth_images[off];
        if (iu_read_exr_from_file(depth_filename, &depth_spec, &output) != SUCCESS) {
            gm_throw(data->log, err, "Failed to read image '%s'\n", depth_filename);
            return false;
        }
    }

    return true;
}

static bool
load_frame_joints_foreach_cb(struct gm_data_index* data_index,
                             int index,
                             const char* path,
                             void* user_data,
                             char** err)
{
    TrainData* data = (TrainData*)user_data;
    const char* top_dir = gm_data_index_get_top_dir(data_index);

    char jnt_filename[512];
    xsnprintf(jnt_filename, sizeof(jnt_filename), "%s/labels/%s.jnt", top_dir, path);

    FILE* fp;
    if (!(fp = fopen(jnt_filename, "rb"))) {
        gm_throw(data->log, err, "Error opening joint file '%s'\n", jnt_filename);
        return false;
    }

    if (fseek(fp, 0, SEEK_END) == -1)
    {
        gm_throw(data->log, err, "Error seeking to end of joint file '%s'\n",
                 jnt_filename);
        return false;
    }

    long n_bytes = ftell(fp);
    if (n_bytes == 0 ||
        n_bytes % sizeof(float) != 0 ||
        (n_bytes % sizeof(float)) % 3 != 0)
    {
        gm_throw(data->log, err, "Unexpected joint file size in '%s'\n",
                 jnt_filename);
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
                 "Error seeking to start of joint file '%s'\n", jnt_filename);
        return false;
    }

    int64_t off = (int64_t)index * n_joints * 3;
    float* joints = &data->joint_data[off];
    if (fread(joints, sizeof(float) * 3, n_joints, fp) != n_joints)
    {
        gm_throw(data->log, err, "%s: Error reading joints\n", jnt_filename);
        return false;
    }

    fclose(fp);

    return true;
}

JSON_Value*
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
                  char** err)
{
    TrainData data = {};
    data.log = log;
    data.gather_depth = !!out_depth_images;
    data.gather_label = !!out_label_images;

    struct gm_data_index* data_index =
        gm_data_index_open(log,
                           data_dir,
                           index_name,
                           err);
    if (!data_index)
        return NULL;

    JSON_Value* meta = json_value_deep_copy(gm_data_index_get_meta(data_index));

    data.n_images = gm_data_index_get_len(data_index);
    data.n_labels = gm_data_index_get_n_labels(data_index);
    data.width = gm_data_index_get_width(data_index);
    data.height = gm_data_index_get_height(data_index);

    /* TODO: consider this per-frame state */
    data.vertical_fov = gm_data_index_get_vfov(data_index);

    size_t n_pixels = (size_t)data.width * data.height * data.n_images;

    if (data.gather_label)
        data.label_images = (uint8_t*)xmalloc(n_pixels * sizeof(uint8_t));

    if (data.gather_depth)
        data.depth_images = (half*)xmalloc(n_pixels * sizeof(half));

    *out_n_images = data.n_images;
    gm_info(log, "Processing %d training images...\n", *out_n_images);

    if (!gm_data_index_foreach(data_index,
                               load_frame_foreach_cb,
                               &data,
                               err))
    {
        goto error;
    }

    /* Handle joints separately since the plan is to factor this out */
    if (out_joints) {
        gm_assert(log, joint_map_path != NULL,
                  "Expected joint_map_path when requesting joint data");

        JSON_Value *map = json_parse_file(joint_map_path);
        if (!map) {
            gm_throw(log, err, "Failed to parse joint map %s\n", joint_map_path);
            goto error;
        }

        /* For now we just care about how many joints there are but maybe
         * we should be handing the map back to the caller somehow?
         */
        data.n_joints = json_array_get_count(json_array(map));
        json_value_free(map);
        map = NULL;

        data.joint_data = (float*)xmalloc(data.n_images * data.n_joints *
                                          3 * sizeof(float));

        if (!gm_data_index_foreach(data_index,
                                   load_frame_joints_foreach_cb,
                                   &data,
                                   err))
        {
            goto error;
        }

        if (out_n_joints)
            *out_n_joints = data.n_joints;
        if (out_joints)
            *out_joints = data.joint_data;
    }

    gm_data_index_destroy(data_index);
    data_index = NULL;

    if (out_width)
        *out_width = data.width;
    if (out_height)
        *out_height = data.height;
    if (out_depth_images)
        *out_depth_images = data.depth_images;
    if (out_label_images)
        *out_label_images = data.label_images;

    return meta;

error:
    xfree(data.label_images);
    xfree(data.depth_images);
    xfree(data.joint_data);
    json_value_free(meta);
    if (data_index)
        gm_data_index_destroy(data_index);

    return NULL;
}

