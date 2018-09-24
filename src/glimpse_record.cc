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
#include <time.h>
#include <list>

#include "glimpse_record.h"
#include "image_utils.h"

#include "parson.h"

#define DEPTH_PATH "/depth"
#define VIDEO_PATH "/video"
#define DEPTH_SUFFIX "-depth.bin"
#define VIDEO_SUFFIX "-video.bin"

struct gm_recording {
    struct gm_logger *log;
    JSON_Value *json;
    JSON_Value *frames;
    int n_frames;
    enum gm_format depth_format;
    enum gm_format video_format;
    char *path;
};

static JSON_Value *
get_json_intrinsics(const struct gm_intrinsics *intrinsics)
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
    json_object_set_number(json_object(json_intrinsics),
                           "distortion_model",
                           intrinsics->distortion_model);
    int n_params = 0;
    switch (intrinsics->distortion_model) {
    case GM_DISTORTION_NONE:
        break;
    case GM_DISTORTION_FOV_MODEL:
        n_params = 1;
        break;
    case GM_DISTORTION_BROWN_K1_K2:
        n_params = 2;
        break;
    case GM_DISTORTION_BROWN_K1_K2_K3:
        n_params = 3;
        break;
    case GM_DISTORTION_BROWN_K1_K2_P1_P2_K3:
        n_params = 5;
        break;
    }

    if (n_params) {
        JSON_Value *params_val = json_value_init_array();
        for (int i = 0; i < n_params; i++) {
            json_array_append_number(json_array(params_val),
                                     intrinsics->distortion[i]);
        }
        json_object_set_value(json_object(json_intrinsics),
                              "distortion_coefficients",
                              params_val);
    }

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
write_bin(struct gm_logger *log, const char *path,
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

struct gm_recording *
gm_recording_init(struct gm_logger *log,
                  struct gm_device *device,
                  const char *recordings_path,
                  const char *rel_path,
                  bool overwrite)
{
    char full_path[512];

    int ret = mkdir(recordings_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Failed to ensure top-level directory exists for recordings");
        return NULL;
    }

    if (overwrite) {
        if (snprintf(full_path, sizeof(full_path), "%s/%s",
                     recordings_path, rel_path) >= (int)sizeof(full_path))
        {
            gm_error(log, "Unable to format recording path");
            return NULL;
        }
    } else {
        time_t unix_time = time(NULL);
        struct tm cur_time = *localtime(&unix_time);
        asctime(&cur_time);

        if (snprintf(full_path, sizeof(full_path), "%s/%d-%d-%d-%d-%d-%d",
                     recordings_path,
                     (int)cur_time.tm_year + 1900,
                     (int)cur_time.tm_mon + 1,
                     (int)cur_time.tm_mday,
                     (int)cur_time.tm_hour,
                     (int)cur_time.tm_min,
                     (int)cur_time.tm_sec) >= (int)sizeof(full_path))
        {
            gm_error(log, "Unable to format recording path");
            return NULL;
        }

        ret = mkdir(full_path, 0777);
        if (ret < 0) {
            gm_error(log, "Failed to create directory for recording");
            return NULL;
        }
    }

    int full_path_len = strlen(full_path);

    // Create depth images directory
    const char *depth_path_suffix = "/depth";
    size_t depth_path_len = full_path_len + strlen(depth_path_suffix);
    char *depth_path = (char *)alloca(depth_path_len + 1);
    snprintf(depth_path, depth_path_len + 1, "%s%s", full_path, depth_path_suffix);

    ret = mkdir(depth_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 depth_path, strerror(errno));
        return nullptr;
    }

    // Create video images directory
    const char *video_path_suffix = "/video";
    size_t video_path_len = full_path_len + strlen(video_path_suffix);
    char *video_path = (char *)alloca(video_path_len + 1);
    snprintf(video_path, video_path_len+1, "%s%s", full_path, video_path_suffix);

    ret = mkdir(video_path, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Unable to create directory '%s': %s",
                 video_path, strerror(errno));
        return nullptr;
    }

    if (overwrite) {
        // Delete any existing files in the depth/video images directory to avoid
        // accumulating untracked files
        delete_files(log, depth_path, DEPTH_SUFFIX);
        delete_files(log, video_path, VIDEO_SUFFIX);
    }

    // Create JSON metadata structure
    JSON_Value *json = json_value_init_object();

    int max_depth_pixels = gm_device_get_max_depth_pixels(device);
    json_object_set_number(json_object(json), "max_depth_pixels", max_depth_pixels);

    int max_video_pixels = gm_device_get_max_video_pixels(device);
    json_object_set_number(json_object(json), "max_video_pixels", max_video_pixels);

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

    // Create an array for frames
    JSON_Value *frames = json_value_init_array();
    json_object_set_value(json_object(json), "frames", frames);

    // Initialise recording structure and return
    struct gm_recording *r = (struct gm_recording *)
        calloc(1, sizeof(struct gm_recording));
    r->log = log;
    r->json = json;
    r->frames = frames;
    r->depth_format = depth_format;
    r->video_format = video_format;
    r->path = strdup(full_path);

    return r;
}

void
gm_recording_save_frame(struct gm_recording *r, struct gm_frame *frame)
{
    if (!frame->video && !frame->depth) {
        gm_warn(r->log, "Not saving frame with no depth or video buffer");
        return;
    }

    JSON_Value *frame_meta = json_value_init_object();

    // Write out frame timestamp
    json_object_set_number(json_object(frame_meta), "timestamp",
                           (double)frame->timestamp);

    if (frame->gravity_valid) {
        JSON_Value *gravity = json_value_init_array();
        for (int i = 0; i < 3; i++)
            json_array_append_number(json_array(gravity), frame->gravity[i]);
        json_object_set_value(json_object(frame_meta), "gravity", gravity);
    }

    if (frame->pose.type != GM_POSE_INVALID) {
        // Write out frame pose data
        JSON_Value *pose = json_value_init_object();
        JSON_Value *orientation = json_value_init_array();
        for (int i = 0; i < 4; ++i) {
            json_array_append_number(json_array(orientation),
                                     (double)frame->pose.orientation[i]);
        }
        JSON_Value *translation = json_value_init_array();
        for (int i = 0; i < 3; ++i) {
            json_array_append_number(json_array(translation),
                                     (double)frame->pose.translation[i]);
        }
        json_object_set_value(json_object(pose), "orientation", orientation);
        json_object_set_value(json_object(pose), "translation", translation);
        json_object_set_number(json_object(pose), "type",
                               (double)frame->pose.type);

        json_object_set_value(json_object(frame_meta), "pose", pose);
    }

    // Write out depth/video frames
    size_t path_len = strlen(r->path);

    if (frame->depth) {
        // Update depth format
        bool save = true;
        if (r->depth_format == GM_FORMAT_UNKNOWN) {
            r->depth_format = frame->depth_format;
            json_object_set_number(json_object(r->json), "depth_format",
                                   (double)r->depth_format);
        } else if (frame->depth_format != r->depth_format) {
            gm_error(r->log, "Depth frame with unexpected format");
            save = false;
        }

        if (save) {
            // Save out depth frame
            // 6 characters: 1 = '/', '4' = %04d, '1' = '\0'
            size_t bin_path_size =
                path_len + strlen(DEPTH_PATH) + strlen(DEPTH_SUFFIX) + 6;
            char *bin_path = (char *)malloc(bin_path_size);
            snprintf(bin_path, bin_path_size, "%s%s/%04d%s",
                     r->path, DEPTH_PATH, r->n_frames, DEPTH_SUFFIX);

            write_bin(r->log, bin_path, frame->depth->data,
                      frame->depth->len);

            json_object_set_string(json_object(frame_meta), "depth_file",
                                   bin_path + path_len);
            json_object_set_number(json_object(frame_meta), "depth_len",
                                   (double)frame->depth->len);
            free(bin_path);

            const struct gm_intrinsics *depth_intrinsics =
                &frame->depth_intrinsics;
            json_object_set_value(json_object(frame_meta), "depth_intrinsics",
                                  get_json_intrinsics(depth_intrinsics));
        }
    }

    if (frame->video) {
        // Update video format
        bool save = true;
        if (r->video_format == GM_FORMAT_UNKNOWN) {
            r->video_format = frame->video_format;
            json_object_set_number(json_object(r->json), "video_format",
                                   (double)r->video_format);
        } else if (frame->video_format != r->video_format) {
            gm_error(r->log, "Video frame with unexpected format");
            save = false;
        }

        if (save) {
            // Save out video frame
            size_t bin_path_size =
                path_len + strlen(VIDEO_PATH) + strlen(VIDEO_SUFFIX) + 6;
            char *bin_path = (char *)malloc(bin_path_size);
            snprintf(bin_path, bin_path_size, "%s%s/%04d%s",
                     r->path, VIDEO_PATH, r->n_frames, VIDEO_SUFFIX);

            write_bin(r->log, bin_path, frame->video->data,
                      frame->video->len);

            json_object_set_string(json_object(frame_meta), "video_file",
                                   bin_path + path_len);
            json_object_set_number(json_object(frame_meta), "video_len",
                                   (double)frame->video->len);
            free(bin_path);

            const struct gm_intrinsics *video_intrinsics =
                &frame->video_intrinsics;
            json_object_set_value(json_object(frame_meta), "video_intrinsics",
                                  get_json_intrinsics(video_intrinsics));
        }
    }

    // Save out camera rotation
    json_object_set_number(json_object(frame_meta), "camera_rotation",
                           (double)frame->camera_rotation);

    json_array_append_value(json_array(r->frames), frame_meta);
    ++r->n_frames;
}

void
gm_recording_close(struct gm_recording *r)
{
    const char *json_name = "glimpse_recording.json";
    size_t json_path_size = strlen(r->path) + strlen(json_name) + 2;
    char *json_path = (char *)alloca(json_path_size);
    snprintf(json_path, json_path_size, "%s/%s", r->path, json_name);

    json_serialize_to_file_pretty(r->json, json_path);

    json_value_free(r->json);
    free(r->path);
    free(r);
}
