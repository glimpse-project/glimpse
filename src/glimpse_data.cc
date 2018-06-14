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

#include "image_utils.h"
#include "xalloc.h"
#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_data.h"

using half_float::half;

#define xsnprintf(dest, size, fmt, ...) do { \
        if (snprintf(dest, size, fmt,  __VA_ARGS__) >= (int)size) \
            exit(1); \
    } while(0)

typedef struct {
    struct gm_logger *log;

    int      n_images;      // Number of training images
    int      n_labels;

    int      width;
    int      height;

    half*    depth_images;  // Depth image data
    uint8_t* label_images;  // Label image data

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

const char *
gm_data_index_get_frame_path(struct gm_data_index* data_index, int n)
{
    if (n < data_index->paths.size())
        return data_index->paths[n];
    else
        return NULL;
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

struct joint_mapping {
    char *name;
    const char *end; // "head" or "tail"
};

struct joints_loader {
    std::vector<joint_mapping> joint_map;
    int n_joints;
    float* joint_data;
};

static bool
load_joints_foreach_cb(struct gm_data_index* data_index,
                       int index,
                       const char* path,
                       void* user_data,
                       char** err)
{
    struct joints_loader* loader = (struct joints_loader*)user_data;
    const char* top_dir = gm_data_index_get_top_dir(data_index);
    std::vector<joint_mapping> &joint_map = loader->joint_map;

    char json_filename[512];
    xsnprintf(json_filename, sizeof(json_filename), "%s/labels/%s.json", top_dir, path);

    JSON_Value *frame_js = json_parse_file(json_filename);
    if (!frame_js) {
        gm_throw(data_index->log, err, "Failed to parse %s", json_filename);
        return false;
    }

    JSON_Array *bones = json_object_get_array(json_object(frame_js), "bones");
    int n_bones = json_array_get_count(bones);

    int n_joints = loader->n_joints;
    float jnt_data[n_joints * 3];

    for (int i = 0; i < n_joints; i++) {
        struct joint_mapping joint = joint_map[i];
        int jnt_pos = i * 3;

        bool found = false;
        for (int j = 0; j < n_bones; j++) {
            JSON_Object *bone = json_array_get_object(bones, j);
            const char *bone_name = json_object_get_string(bone, "name");
            if (strcmp(joint.name, bone_name) != 0)
                continue;

            JSON_Array *end = json_object_get_array(bone, joint.end);
            if (!end)
                break;

            jnt_data[jnt_pos+0] = json_array_get_number(end, 0);
            jnt_data[jnt_pos+1] = json_array_get_number(end, 1);
            jnt_data[jnt_pos+2] = json_array_get_number(end, 2);

            //printf("%s.%s = (%5.2f, %5.2f, %5.2f)\n",
            //       joint.name, joint.end,
            //       jnt_data[jnt_pos+0],
            //       jnt_data[jnt_pos+1],
            //       jnt_data[jnt_pos+2]);
            found  = true;
            break;
        }

        if (!found) {
            gm_throw(data_index->log, err, "Failed to find bone %s.%s in %s",
                     joint.name, joint.end, json_filename);
            json_value_free(frame_js);
            return false;
        }
    }

    int64_t off = (int64_t)index * n_joints * 3;
    float* joints = &loader->joint_data[off];
    memcpy(joints, jnt_data, sizeof(jnt_data));

    json_value_free(frame_js);
    return true;
}

bool
gm_data_index_load_joints(struct gm_data_index* data_index,
                          const char* joint_map_file,
                          int* out_n_joints,
                          float** out_joints,
                          char** err)
{
    struct joints_loader loader = {};
    int n_images = gm_data_index_get_len(data_index);
    bool status = false;

    gm_assert(data_index->log, out_n_joints != NULL, "Must pass out_n_joints pointer");
    gm_assert(data_index->log, out_joints != NULL, "Must pass out_joints pointer");

    JSON_Value *joint_map_val = json_parse_file(joint_map_file);
    if (!joint_map_val) {
        gm_throw(data_index->log, err, "Failed to parse %s", joint_map_file);
        return false;
    }
    JSON_Array *joint_map = json_array(joint_map_val);

    int n_joints = json_array_get_count(joint_map);
    loader.n_joints = n_joints;

    for (int i = 0; i < n_joints; i++) {
        JSON_Object *joint = json_array_get_object(joint_map, i);
        struct joint_mapping mapping;
        int name_len;

        const char *name = json_object_get_string(joint, "joint");
        const char *dot = strstr(name, ".");
        if (!dot) {
            gm_throw(data_index->log, err,
                     "Spurious joint %s in %s not formatted like <name>.<end>",
                     name, joint_map_file);
            goto exit;
        }
        name_len = dot - name;
        mapping.name = strndup(name, name_len);
        mapping.end = name + name_len + 1;
        loader.joint_map.push_back(mapping);
    }

    loader.joint_data = (float*)xmalloc(n_images * n_joints * 3 * sizeof(float));

    if (!gm_data_index_foreach(data_index,
                               load_joints_foreach_cb,
                               &loader, // user data
                               err))
    {
        xfree(loader.joint_data);
        goto exit;
    }

    *out_n_joints = n_joints;
    *out_joints = loader.joint_data;

    status = true;
exit:

    // NB: might have failed part-way through parsing joint map so size might
    // not == n_joints
    for (int i = 0; i < (int)loader.joint_map.size(); i++) {
        free(loader.joint_map[i].name);
        // don't need to free .end which points to const string within json state
    }

    json_value_free(joint_map_val);
    return status;
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

JSON_Value*
gm_data_load_simple(struct gm_logger* log,
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

        if (!gm_data_index_load_joints(data_index,
                                       joint_map_path,
                                       out_n_joints,
                                       out_joints,
                                       err))
        {
            goto error;
        }
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
    json_value_free(meta);
    if (data_index)
        gm_data_index_destroy(data_index);

    return NULL;
}

JSON_Value*
gm_data_load_label_map_from_json(struct gm_logger* log,
                                 const char* filename,
                                 uint8_t* map,
                                 char** err)
{
    JSON_Value* label_map = json_parse_file(filename);
    if (!label_map) {
        gm_throw(log, err, "Failed to parse label map %s", filename);
        return NULL;
    }

    memset(map, 0, 256);

    JSON_Array* label_map_array = json_array(label_map);
    for (int i = 0; i < (int)json_array_get_count(label_map_array); i++) {
        JSON_Object* mapping = json_array_get_object(label_map_array, i);
        const char* label_name = json_object_get_string(mapping, "name");

        JSON_Array* inputs = json_object_get_array(mapping, "inputs");
        for (int j = 0; j < (int)json_array_get_count(inputs); j++) {
            int input = json_array_get_number(inputs, j);
            if (input < 0 || input > 255) {
                gm_throw(log, err, "Out of range \"%s\" label mapping from %d in %s\n",
                         label_name, input, filename);
                json_value_free(label_map);
                return NULL;
            }
            if (map[input]) {
                gm_throw(log, err, "Input %d sampled by multiple labels in %s\n",
                         input, filename);
                json_value_free(label_map);
                return NULL;
            }
            map[input] = i;
        }
    }

    return label_map;
}
