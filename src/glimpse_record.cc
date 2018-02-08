/*
 * Copyright (C) 2018 Glimp IP Ltd
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

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <alloca.h>
#include <list>

#include "glimpse_record.h"
#include "image_utils.h"

#include "parson.h"

static JSON_Value *
gm_record_get_json_intrinsics(const struct gm_intrinsics *intrinsics)
{
    JSON_Value *json_intrinsics = json_value_init_object();
    json_object_set_number(json_object(json_intrinsics), "width",
                           (double)intrinsics->width);
    json_object_set_number(json_object(json_intrinsics), "height",
                           (double)intrinsics->height);
    json_object_set_number(json_object(json_intrinsics), "fx", intrinsics->fx);
    json_object_set_number(json_object(json_intrinsics), "fy", intrinsics->fy);
    json_object_set_number(json_object(json_intrinsics), "cx", intrinsics->cx);
    json_object_set_number(json_object(json_intrinsics), "cy", intrinsics->cy);
    return json_intrinsics;
}

static void
delete_files(struct gm_logger *log, const char *path, const char *suffix)
{
    DIR *dir = opendir(path);
    if (!dir) {
        gm_error(log, "Error opening path %s: %s", path, strerror(errno));
        return;
    }

    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) {
            continue;
        }

        size_t file_len = strlen(entry->d_name);
        if (file_len <= suffix_len) {
            continue;
        }

        char *ext = (entry->d_name + strlen(entry->d_name)) - strlen(suffix);
        if (strcmp(ext, suffix) != 0) {
            continue;
        }

        // The entry is a regular file and matches the suffix, delete it
        size_t full_path_size = path_len + file_len + 2;
        char *full_path = (char *)malloc(full_path_size);
        snprintf(full_path, full_path_size, "%s/%s", path, entry->d_name);
        if (remove(full_path) != 0) {
            gm_warn(log, "Error removing file %s: %s",
                    full_path, strerror(errno));
        }
        free(full_path);
    }
}

void
gm_record_save(struct gm_logger *log, struct gm_device *device,
               const std::list<struct gm_tracking *> &record, const char *path)
{
    // Create the directory structure for the recording
    int ret = mkdir(path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 path, strerror(errno));
        return;
    }

    // Create depth images directory
    const char *depth_suffix = "/depth";
    size_t path_len = strlen(path);
    size_t depth_path_len = path_len + strlen(depth_suffix);
    char *depth_path = (char *)alloca(depth_path_len + 1);
    snprintf(depth_path, depth_path_len + 1, "%s%s", path, depth_suffix);

    ret = mkdir(depth_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 depth_path, strerror(errno));
        return;
    }

    // Create video images directory
    const char *video_suffix = "/video";
    size_t video_path_len = path_len + strlen(video_suffix);
    char *video_path = (char *)alloca(video_path_len + 1);
    snprintf(video_path, video_path_len+1, "%s%s", path, video_suffix);

    ret = mkdir(video_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 video_path, strerror(errno));
        return;
    }

    // Delete any existing files in the depth/video images directory to avoid
    // accumulating untracked files
    const char *exr_suffix = ".exr";
    const char *png_suffix = ".png";
    delete_files(log, depth_path, exr_suffix);
    delete_files(log, video_path, png_suffix);

    // Create JSON metadata structure
    JSON_Value *json = json_value_init_object();

    // Save depth intrinsics
    const struct gm_intrinsics *depth_intrinsics =
      gm_device_get_depth_intrinsics(device);
    json_object_set_value(json_object(json), "depth_intrinsics",
                          gm_record_get_json_intrinsics(depth_intrinsics));

    // Save video intrinsics
    const struct gm_intrinsics *video_intrinsics =
      gm_device_get_video_intrinsics(device);
    json_object_set_value(json_object(json), "video_intrinsics",
                          gm_record_get_json_intrinsics(video_intrinsics));

    // Save depth-to-video extrinsics
    struct gm_extrinsics *extrinsics =
        gm_device_get_depth_to_video_extrinsics(device);
    JSON_Value *json_extrinsics = json_value_init_object();

    JSON_Value *rotation = json_value_init_array();
    for (int i = 0; i < 9; ++i) {
        json_array_append_number(json_array(rotation),
                                 (double)extrinsics->rotation[i]);
    }
    json_object_set_value(json_object(json_extrinsics), "rotation", rotation);

    JSON_Value *translation = json_value_init_array();
    for (int i = 0; i < 3; ++i) {
        json_array_append_number(json_array(translation),
                                 (double)extrinsics->translation[i]);
    }
    json_object_set_value(json_object(json_extrinsics), "translation",
                          translation);

    json_object_set_value(json_object(json), "depth_to_video_extrinsics",
                          json_extrinsics);

    // Save out depth/video frames and metadata
    JSON_Value *frames = json_value_init_array();
    size_t exr_suffix_len = strlen(exr_suffix);
    size_t png_suffix_len = strlen(png_suffix);
    int i = 0;
    for (std::list<struct gm_tracking *>::const_iterator it = record.begin();
         it != record.end(); ++it, ++i) {
        // Save out depth frame
        IUImageSpec spec = {
            (int)depth_intrinsics->width,
            (int)depth_intrinsics->height,
            IU_FORMAT_FLOAT
        };

        // 6 characters: 1 = '/', '4' = %04d, '1' = '\0'
        size_t exr_path_size = depth_path_len + exr_suffix_len + 6;
        char *exr_path = (char *)malloc(exr_path_size);
        snprintf(exr_path, exr_path_size, "%s/%04d%s",
                 depth_path, i, exr_suffix);

        IUReturnCode ret =
            iu_write_exr_to_file(exr_path, &spec,
                                 (void *)gm_tracking_get_depth(*it),
                                 IU_FORMAT_FLOAT);
        if (ret != SUCCESS) {
            gm_error(log, "Error writing '%s' (%d)", exr_path, ret);
        }

        // Save out video frame
        spec.format = IU_FORMAT_U32;
        size_t png_path_size = video_path_len + png_suffix_len + 6;
        char *png_path = (char *)malloc(png_path_size);
        snprintf(png_path, png_path_size, "%s/%04d%s",
                 video_path, i, png_suffix);

        ret = iu_write_png_to_file(png_path, &spec,
                                   (void *)gm_tracking_get_video(*it),
                                   nullptr, 0);
        if (ret != SUCCESS) {
            gm_error(log, "Error writing '%s' (%d)", png_path, ret);
        }

        // Write frame metadata to JSON structure
        JSON_Value *frame_meta = json_value_init_object();
        json_object_set_number(json_object(frame_meta), "depth_timestamp",
                               (double)gm_tracking_get_depth_timestamp(*it));
        json_object_set_number(json_object(frame_meta), "video_timestamp",
                               (double)gm_tracking_get_video_timestamp(*it));
        json_object_set_string(json_object(frame_meta), "depth_file", exr_path);
        json_object_set_string(json_object(frame_meta), "video_file", png_path);
        json_array_append_value(json_array(frames), frame_meta);

        free(exr_path);
        free(png_path);
    }
    json_object_set_value(json_object(json), "frames", frames);

    // Write to file, free and return
    const char *json_name = "glimpse_recording.json";
    size_t json_path_size = path_len + strlen(json_name) + 2;
    char *json_path = (char *)alloca(json_path_size);
    snprintf(json_path, json_path_size, "%s/%s", path, json_name);
    json_serialize_to_file_pretty(json, json_path);
    json_value_free(json);
}
