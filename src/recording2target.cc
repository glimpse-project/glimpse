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

#include <getopt.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "parson.h"

#include "glimpse_assets.h"
#include "glimpse_context.h"
#include "glimpse_device.h"
#include "glimpse_log.h"

typedef struct {
    struct gm_logger *log;
    struct gm_context *ctx;
    struct gm_device *device;
    struct gm_frame *last_depth_frame;
    struct gm_frame *last_video_frame;
    const char *out_dir;
    FILE *index;
    pthread_mutex_t finished_cond_mutex;
    pthread_cond_t finished_cond;
    bool finished;
    struct gm_ui_property *frame_property;
    int last_frame;
    int tracked_frame;
    int end_frame;
    uint64_t tracked_time;
    uint64_t last_time;
    uint64_t frame_time;
} Data;

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: recording2target [OPTIONS] <recording directory> <output directory>\n"
"Using a video/depth recording sequence, render a motion target sequence.\n"
"\n"
"  -c, --config=FILE      Use this particular Glimpse device config\n"
"  -b, --begin=NUMBER     Begin on n frame (default: 1)\n"
"  -e, --end=NUMBER       End on this frame (default: unset)\n"
"  -t, --time=NUMBER      Minimum number of seconds between frames (default: 0)\n"
"  -v, --verbose          Verbose output\n"
"  -h, --help             Display this help\n\n");
}

static bool
check_complete(Data *data)
{
    if (data->last_frame >= data->frame_property->int_state.max ||
        (data->end_frame && (data->last_frame >= data->end_frame))) {
        pthread_mutex_lock(&data->finished_cond_mutex);
        data->finished = true;
        pthread_cond_signal(&data->finished_cond);
        pthread_mutex_unlock(&data->finished_cond_mutex);
    }

    return data->finished;
}

static void
on_event_cb(struct gm_context *ctx, struct gm_event *event, void *user_data)
{
    struct gm_tracking *tracking;
    Data *data = (Data *)user_data;

    switch (event->type) {
    case GM_EVENT_REQUEST_FRAME:
        gm_debug(data->log, "Request for frame");
        gm_device_request_frame(data->device, GM_REQUEST_FRAME_DEPTH |
                                              GM_REQUEST_FRAME_VIDEO);
        break;
    case GM_EVENT_TRACKING_READY:
        // FIXME: I think there's a possible race here - This event gets sent
        //        before the request_frame event, but on a separate thread.
        //        Though massively unlikely, I suppose it could be possible
        //        for the request to be completely handled and data->last_frame
        //        to be reset before reaching this assignment.
        int frame = data->last_frame;
        uint64_t frame_time = data->tracked_time;

        gm_message(data->log, "Processing frame %d/%d",
                   frame, data->frame_property->int_state.max);

        // Save out skeleton if enough time has elapsed since the last tracked
        // frame.
        tracking = gm_context_get_latest_tracking(data->ctx);
        if (tracking && gm_tracking_was_successful(tracking) &&
            (data->last_time == 0 ||
             data->last_time + data->frame_time <= frame_time))
        {
            data->last_time = frame_time;

            const struct gm_skeleton *skeleton =
                gm_tracking_get_skeleton(tracking);

            JSON_Value *root = json_value_init_object();
            JSON_Value *bones = json_value_init_array();
            json_object_set_value(json_object(root), "bones", bones);

            for (int j = 0; j < gm_skeleton_get_n_joints(skeleton); ++j) {
                const struct gm_joint *joint =
                    gm_skeleton_get_joint(skeleton, j);
                char *bone_name = strdup(joint->name);
                char *bone_part = strchr(bone_name, (int)'.');
                if (bone_part) {
                    bone_part[0] = '\0';
                    ++bone_part;

                    // Find bone, or create one if this is the first encounter
                    JSON_Value *bone = NULL;
                    for (int c = 0;
                         c < json_array_get_count(json_array(bones)); ++c)
                    {
                        JSON_Value *bone_obj =
                            json_array_get_value(json_array(bones), c);
                        if (strcmp(json_object_get_string(json_object(bone_obj),
                                                          "name"),
                                   bone_name) == 0)
                        {
                            bone = bone_obj;
                            break;
                        }
                    }
                    if (!bone) {
                        bone = json_value_init_object();
                        json_object_set_string(json_object(bone), "name",
                                               bone_name);
                        json_array_append_value(json_array(bones), bone);
                    }

                    JSON_Value *joint_array = json_value_init_array();
                    json_object_set_value(json_object(bone), bone_part,
                                          joint_array);
                    json_array_append_number(json_array(joint_array), joint->x);
                    json_array_append_number(json_array(joint_array), joint->y);
                    json_array_append_number(json_array(joint_array), joint->z);
                }
                free(bone_name);
            }

            char output_name[1024];
            snprintf(output_name, 1024, "%s/%06d.json", data->out_dir,
                     frame);
            json_serialize_to_file_pretty(root, output_name);

            json_value_free(root);

            // Add file to index
            snprintf(output_name, 1024, "%06d.json\n", frame);
            fputs(output_name, data->index);
        }
        if (tracking) {
            gm_tracking_unref(tracking);
        }

        // Signal finishing if this was the last frame
        check_complete(data);
        break;
    }

    gm_context_event_free(event);
}

static void
on_device_event_cb(struct gm_device_event *event, void *user_data)
{
    Data *data = (Data *)user_data;

    struct gm_frame *frame;
    struct gm_ui_properties *props;
    int max_depth_pixels, max_video_pixels;

    switch (event->type) {
    case GM_DEV_EVENT_READY:
        gm_debug(data->log, "Device ready");
        max_depth_pixels = gm_device_get_max_depth_pixels(data->device);
        gm_context_set_max_depth_pixels(data->ctx, max_depth_pixels);

        max_video_pixels = gm_device_get_max_video_pixels(data->device);
        gm_context_set_max_video_pixels(data->ctx, max_video_pixels);

        props = gm_device_get_ui_properties(data->device);
        gm_props_set_bool(props, "loop", false);
        data->frame_property = gm_props_lookup(props, "frame");

        gm_device_start(data->device);
        gm_context_enable(data->ctx);

        break;

    case GM_DEV_EVENT_FRAME_READY:
        gm_debug(data->log, "Frame ready");
        frame = gm_device_get_latest_frame(data->device);
        if (frame->depth) {
            if (data->last_depth_frame) {
                gm_frame_unref(data->last_depth_frame);
            }
            data->last_depth_frame = gm_frame_ref(frame);
        }
        if (frame->video) {
            if (data->last_video_frame) {
                gm_frame_unref(data->last_video_frame);
            }
            data->last_video_frame = gm_frame_ref(frame);
        }
        gm_frame_unref(frame);

        if (data->last_video_frame && data->last_depth_frame) {
            if (data->last_video_frame != data->last_depth_frame) {
                frame = gm_device_combine_frames(data->device,
                                                 data->last_depth_frame,
                                                 data->last_depth_frame,
                                                 data->last_video_frame);
            } else {
                frame = gm_frame_ref(data->last_depth_frame);
            }

            bool tracking = false;
            int n_frame = gm_prop_get_int(data->frame_property);
            if (n_frame > data->last_frame) {
                gm_debug(data->log, "Sending frame to context");
                data->last_frame = n_frame;
                data->tracked_time = frame->timestamp;
                gm_context_notify_frame(data->ctx, frame);
                tracking = true;
            } else {
                gm_debug(data->log, "Skipping frame");
            }
            gm_frame_unref(frame);

            if (data->last_depth_frame) {
                gm_frame_unref(data->last_depth_frame);
                data->last_depth_frame = NULL;
            }
            if (data->last_video_frame) {
                gm_frame_unref(data->last_video_frame);
                data->last_video_frame = NULL;
            }

            if (!tracking) {
                if (!check_complete(data)) {
                    // If we're not tracking this frame and we aren't finished
                    // tracking entirely, request another frame.
                    gm_debug(data->log, "Requesting frame");
                    gm_device_request_frame(data->device,
                                            GM_REQUEST_FRAME_DEPTH |
                                            GM_REQUEST_FRAME_VIDEO);
                }
            }
        }

        break;
    }

    gm_device_event_free(event);
}

int
main(int argc, char **argv)
{
    bool verbose_output = false;

    const char *short_opts = "+c:b:e:t:vh";
    const struct option long_opts[] = {
        {"config",          required_argument,  0, 'c'},
        {"begin",           required_argument,  0, 'b'},
        {"end",             required_argument,  0, 'e'},
        {"time",            required_argument,  0, 't'},
        {"verbose",         no_argument,        0, 'v'},
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    Data data;
    memset(&data, 0, sizeof(Data));

    int opt;
    const char *config_filename = NULL;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'c':
            config_filename = optarg;
            break;
        case 'b':
            data.last_frame = atoi(optarg) - 1;
            if (data.last_frame < 0) data.last_frame = 0;
            break;
        case 'e':
            data.end_frame = atoi(optarg);
            if (data.end_frame < 0) data.end_frame = 0;
            break;
        case 't':
            data.frame_time = (uint64_t)(strtod(optarg, NULL) * 1000000000.0);
            break;
        case 'v':
            verbose_output = true;
            break;
        case 'h':
            print_usage(stdout);
            return 0;
        default:
            print_usage(stderr);
            return 1;
        }
    }

    if ((argc - optind) < 2) {
        print_usage(stderr);
        return 1;
    }

    const char *record_dir = argv[optind];

    data.out_dir = argv[optind + 1];
    data.log = gm_logger_new(NULL, (void *)&verbose_output);

    const char *assets_root_env = getenv("GLIMPSE_ASSETS_ROOT");
    char *assets_root = strdup(assets_root_env ? assets_root_env : "");
    gm_set_assets_root(data.log, assets_root);

    gm_debug(data.log, "Creating context");
    data.ctx = gm_context_new(data.log, NULL);
    gm_context_set_event_callback(data.ctx, on_event_cb, &data);

    gm_debug(data.log, "Opening device config");
    if (config_filename) {
        struct stat sb;

        if (stat(config_filename, &sb) != 0) {
            gm_error(data.log, "Failed to stat %s\n", config_filename);
            return 1;
        }

        FILE *config_file = fopen(config_filename, "rb");
        if (!config_file) {
            gm_error(data.log, "Failed to open %s\n", config_filename);
            return 1;
        }

        char *buf = (char *)malloc(sb.st_size);
        if (fread(buf, sb.st_size, 1, config_file) != 1) {
            gm_error(data.log, "Failed to read %s\n", config_filename);
            return 1;
        }
        fclose(config_file);

        JSON_Value *json_props = json_parse_string(buf);
        gm_props_from_json(data.log, gm_context_get_ui_properties(data.ctx),
                           json_props);
        json_value_free(json_props);

        free(buf);
    } else {
        char *open_err = NULL;
        struct gm_asset *config_asset = gm_asset_open(data.log,
                                                      "glimpse-config.json",
                                                      GM_ASSET_MODE_BUFFER,
                                                      &open_err);
        if (config_asset) {
            const char *buf = (const char *)gm_asset_get_buffer(config_asset);
            JSON_Value *json_props = json_parse_string(buf);
            gm_props_from_json(data.log, gm_context_get_ui_properties(data.ctx),
                               json_props);
            json_value_free(json_props);
            gm_asset_close(config_asset);
        } else {
            gm_warn(data.log, "Failed to open glimpse-config.json: %s",
                    open_err);
            free(open_err);
        }
    }

    struct gm_device_config config = {};
    config.type = GM_DEVICE_RECORDING;
    config.recording.path = record_dir;
    config.recording.disable_frame_skip = true;

    pthread_mutex_init(&data.finished_cond_mutex, NULL);
    pthread_cond_init(&data.finished_cond, NULL);
    pthread_mutex_lock(&data.finished_cond_mutex);

    // Check if the output directory exists, and if not, try to make it
    struct stat file_props;
    if (stat(data.out_dir, &file_props) == 0) {
        // If the file exists, make sure it's a directory
        if (!S_ISDIR(file_props.st_mode)) {
            gm_error(data.log,
                     "Output directory '%s' exists but is not a directory",
                     data.out_dir);
            return 1;
        }
    } else {
        // Create the directory
        if (mkdir(data.out_dir, 0755) != 0) {
            gm_error(data.log, "Failed to create output directory");
            return 1;
        }
    }

    // Open the index file
    char index_name[1024];
    snprintf(index_name, 1024, "%s/glimpse_target.index", data.out_dir);
    if (!(data.index = fopen(index_name, "w"))) {
        gm_error(data.log, "Failed to open index file '%s'", index_name);
        return 1;
    }

    gm_debug(data.log, "Opening device");
    data.device = gm_device_open(data.log, &config, NULL);
    gm_device_set_event_callback(data.device, on_device_event_cb, &data);
    gm_debug(data.log, "Committing device config");
    gm_device_commit_config(data.device, NULL);

    gm_debug(data.log, "Waiting...");
    while (!data.finished) {
        pthread_cond_wait(&data.finished_cond,
                          &data.finished_cond_mutex);
    }
    pthread_mutex_unlock(&data.finished_cond_mutex);

    gm_device_stop(data.device);
    gm_context_destroy(data.ctx);

    if (data.last_depth_frame) {
        gm_frame_unref(data.last_depth_frame);
    }
    if (data.last_video_frame) {
        gm_frame_unref(data.last_video_frame);
    }

    gm_device_close(data.device);

    fclose(data.index);
    gm_logger_destroy(data.log);

    return 0;
}
