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

static void
gm_record_write_bin(struct gm_logger *log, const char *path,
                    void *data, size_t len)
{
    FILE *bin_file = fopen(path, "w");
    if (bin_file) {
        if (fwrite(data, 1, len, bin_file) != len) {
            gm_error(log, "Error writing '%s'", path);
        }
    } else {
        gm_error(log, "Error opening '%s': %s", path,
                 strerror(errno));
    }
    if (fclose(bin_file) != 0) {
        gm_error(log, "Error closing '%s': %s", path,
                 strerror(errno));
    }
}

void
gm_record_save(struct gm_logger *log, struct gm_device *device,
               const std::list<struct gm_frame *> &record, const char *path,
               bool overwrite)
{
    // Create the directory structure for the recording
    // If the directory exists, append a number
    int ret;
    int path_suffix = 0;
    size_t path_len = strlen(path);
    char *path_copy = (char *)malloc(strlen(path) + 5);
    strncpy(path_copy, path, path_len + 1);
    do {
        ret = mkdir(path_copy, 0777);
        if (ret < 0) {
            if (errno == EEXIST && path_suffix < 9999) {
                if (overwrite) {
                    break;
                }

                ++path_suffix;
                snprintf(path_copy + path_len, 5, "%04d", path_suffix);
            } else {
                gm_error(log, "Unable to create directory '%s': %s",
                         path, strerror(errno));
                return;
            }
        }
    } while (ret != 0);
    path = path_copy;
    path_len = strlen(path);

    // Create depth images directory
    const char *depth_path_suffix = "/depth";
    size_t depth_path_len = path_len + strlen(depth_path_suffix);
    char *depth_path = (char *)alloca(depth_path_len + 1);
    snprintf(depth_path, depth_path_len + 1, "%s%s", path, depth_path_suffix);

    ret = mkdir(depth_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 depth_path, strerror(errno));
        return;
    }

    // Create video images directory
    const char *video_path_suffix = "/video";
    size_t video_path_len = path_len + strlen(video_path_suffix);
    char *video_path = (char *)alloca(video_path_len + 1);
    snprintf(video_path, video_path_len+1, "%s%s", path, video_path_suffix);

    ret = mkdir(video_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 video_path, strerror(errno));
        return;
    }

    // Delete any existing files in the depth/video images directory to avoid
    // accumulating untracked files
    const char *depth_suffix = "-depth.bin";
    const char *video_suffix = "-video.bin";
    delete_files(log, depth_path, depth_suffix);
    delete_files(log, video_path, video_suffix);

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

    // Save out preliminary format information
    enum gm_format depth_format = GM_FORMAT_UNKNOWN;
    enum gm_format video_format = GM_FORMAT_UNKNOWN;
    json_object_set_number(json_object(json), "depth_format",
                           (double)GM_FORMAT_UNKNOWN);
    json_object_set_number(json_object(json), "video_format",
                           (double)GM_FORMAT_UNKNOWN);

    // Save out depth/video frames and metadata
    JSON_Value *frames = json_value_init_array();
    size_t depth_suffix_len = strlen(depth_suffix);
    size_t video_suffix_len = strlen(video_suffix);
    int i = 0;
    for (std::list<struct gm_frame *>::const_iterator it = record.begin();
         it != record.end(); ++it, ++i) {

        JSON_Value *frame_meta = json_value_init_object();
        json_object_set_number(json_object(frame_meta), "timestamp",
                               (double)(*it)->timestamp);

        if ((*it)->depth) {
            // Update depth format
            if (depth_format == GM_FORMAT_UNKNOWN) {
                depth_format = (*it)->depth_format;
                json_object_set_number(json_object(json), "depth_format",
                                       (double)depth_format);
            } else if ((*it)->depth_format != depth_format) {
                gm_error(log, "Depth frame with unexpected format");
                continue;
            }

            // Save out depth frame
            // 6 characters: 1 = '/', '4' = %04d, '1' = '\0'
            size_t bin_path_size = depth_path_len + depth_suffix_len + 6;
            char *bin_path = (char *)malloc(bin_path_size);
            snprintf(bin_path, bin_path_size, "%s/%04d%s",
                     depth_path, i, depth_suffix);

            gm_record_write_bin(log, bin_path, (*it)->depth->data,
                                (*it)->depth->len);

            json_object_set_string(json_object(frame_meta), "depth_file",
                                   bin_path + path_len);
            json_object_set_number(json_object(frame_meta), "depth_len",
                                   (double)(*it)->depth->len);
            free(bin_path);
        }

        if ((*it)->video) {
            // Update video format
            if (video_format == GM_FORMAT_UNKNOWN) {
                video_format = (*it)->video_format;
                json_object_set_number(json_object(json), "video_format",
                                       (double)video_format);
            } else if ((*it)->video_format != video_format) {
                gm_error(log, "Video frame with unexpected format");
                continue;
            }

            // Save out video frame
            size_t bin_path_size = video_path_len + video_suffix_len + 6;
            char *bin_path = (char *)malloc(bin_path_size);
            snprintf(bin_path, bin_path_size, "%s/%04d%s",
                     video_path, i, video_suffix);

            gm_record_write_bin(log, bin_path, (*it)->video->data,
                                (*it)->video->len);

            json_object_set_string(json_object(frame_meta), "video_file",
                                   bin_path + path_len);
            json_object_set_number(json_object(frame_meta), "video_len",
                                   (double)(*it)->video->len);
            free(bin_path);
        }

        if ((*it)->depth || (*it)->video) {
            json_array_append_value(json_array(frames), frame_meta);
        } else {
            json_value_free(frame_meta);
        }
    }
    json_object_set_value(json_object(json), "frames", frames);

    // Write to file, free and return
    const char *json_name = "glimpse_recording.json";
    size_t json_path_size = path_len + strlen(json_name) + 2;
    char *json_path = (char *)alloca(json_path_size);
    snprintf(json_path, json_path_size, "%s/%s", path, json_name);
    json_serialize_to_file_pretty(json, json_path);
    json_value_free(json);
    free(path_copy);
}
