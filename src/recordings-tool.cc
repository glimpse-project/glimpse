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
#include <inttypes.h>

#include <vector>

#include "parson.h"

#include "glimpse_assets.h"
#include "glimpse_context.h"
#include "glimpse_device.h"
#include "glimpse_log.h"

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "rec-tool"

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

enum event_type
{
    EVENT_DEVICE,
    EVENT_CONTEXT
};

struct event
{
    enum event_type type;
    union {
        struct gm_event *context_event;
        struct gm_device_event *device_event;
    };
};

typedef struct {
    struct gm_logger *log;
    struct gm_context *ctx;

    int command_index;

    char *recording_dir;
    struct gm_device *device;
    struct gm_ui_property *recording_frame_prop;

    /* Events from the gm_context and gm_device apis may be delivered via any
     * arbitrary thread which we don't want to block, and at a time where
     * the gm_ apis may not be reentrant due to locks held during event
     * notification
     */
    pthread_mutex_t event_queue_lock;
    pthread_cond_t event_notify_cond;
    std::vector<struct event> *events_back;
    std::vector<struct event> *events_front;

    /* Set when gm_device sends a _FRAME_READY device event */
    bool device_frame_ready;
    int notified_frame_no;

    /* Once we've been notified that there's a device frame ready for us then
     * we store the latest frames from gm_device_get_latest_frame() here...
     */
    struct gm_frame *last_depth_frame;
    int last_depth_frame_no;
    struct gm_frame *last_video_frame;
    int last_video_frame_no;

    /* Set when gm_context sends a _REQUEST_FRAME event */
    bool context_needs_frame;

    /* Set when gm_context sends a _TRACKING_READY event */
    bool tracking_ready;

    /* Info about the last frame sent to gm_context for tracking (NB: the
     * frame we send for tracking may combine buffers from different
     * recording frames, so we have separate numbers for the depth and
     * video buffers).
     *
     * last_tracking_timestamp == 0 if no frame currently being tracked.
     */
    int last_tracking_frame_depth_no;
    int last_tracking_frame_video_no;
    uint64_t last_tracking_timestamp;

    int begin_frame;
    int end_frame;
    uint64_t time_step;

    bool finished;

    // Command specific state...
    union {
        struct {
            /* Timestamp for the last frame written as a target, used to figure
             * out what should be skipped over if data->time_step requested
             */
            uint64_t last_written_timestamp;

            const char *out_dir;
            FILE *index;
        } make_targets;
    };
} Data;


static void
make_target_sequence_print_usage(FILE* stream)
{
    fprintf(stream,
"Usage: make_target_sequence [options...] <recording directory> <output directory>\n"
"\n"
"Converts tracked recording frames into a sequence of target poses\n"
"\n"
"  -h, --help             Display this help\n"
"\n"
    );
}

static bool
make_target_sequence_argparse(Data *data, int argc, char **argv)
{
    optind = 0; // reset getopt parser state

    const char *short_opts = "h";
    const struct option long_opts[] = {
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'h':
            make_target_sequence_print_usage(stdout);
            return 0;
        default:
            make_target_sequence_print_usage(stderr);
            return 1;
        }
    }

    if (argc != 3) {
        make_target_sequence_print_usage(stderr);
        return false;
    }

    data->recording_dir = strdup(argv[1]);

    data->make_targets.out_dir = argv[2];

    return true;
}

static bool
make_target_sequence_start(Data *data)
{
    // Check if the output directory exists, and if not, try to make it
    struct stat file_props;
    if (stat(data->make_targets.out_dir, &file_props) == 0) {
        // If the file exists, make sure it's a directory
        if (!S_ISDIR(file_props.st_mode)) {
            gm_error(data->log,
                     "Output directory '%s' exists but is not a directory",
                     data->make_targets.out_dir);
            return false;
        }
    } else {
        // Create the directory
        if (mkdir(data->make_targets.out_dir, 0755) != 0) {
            gm_error(data->log, "Failed to create output directory");
            return false;
        }
    }

    // Open the index file
    char index_name[1024];
    snprintf(index_name, 1024, "%s/glimpse_target.index",
             data->make_targets.out_dir);
    if (!(data->make_targets.index = fopen(index_name, "w"))) {
        gm_error(data->log, "Failed to open index file '%s'", index_name);
        return false;
    }

    return true;
}

static bool
append_tracking_target(Data *data,
                       struct gm_tracking *tracking,
                       int recording_frame_no)
{
    const struct gm_skeleton *skeleton = gm_tracking_get_skeleton(tracking);

    char output_name[1024];
    snprintf(output_name, 1024, "%s/%06d.json", data->make_targets.out_dir,
             recording_frame_no);

    if (gm_skeleton_save(skeleton, output_name)) {
        // Add file to index
        snprintf(output_name, 1024, "%06d.json\n", recording_frame_no);
        fputs(output_name, data->make_targets.index);

        data->make_targets.last_written_timestamp = data->last_tracking_timestamp;

        return true;
    }

    return false;
}

static void
make_target_sequence_tracking_ready(Data *data)
{
    int recording_frame_no = data->last_tracking_frame_depth_no;

    uint64_t elapsed = UINT64_MAX;
    if (data->make_targets.last_written_timestamp) {
        elapsed = (data->last_tracking_timestamp -
                   data->make_targets.last_written_timestamp);
    }
    if (elapsed < data->time_step) {
        gm_debug(data->log, "Skipping unwanted recording frame %d, due to time step",
                 recording_frame_no);
        return;
    }

    gm_message(data->log, "Processing frame %d/%d",
               recording_frame_no,
               data->recording_frame_prop->int_state.max);

    struct gm_tracking *tracking = gm_context_get_latest_tracking(data->ctx);
    gm_assert(data->log, tracking != NULL,
              "Spurious NULL tracking after _TRACKING_READY notification");

    if (gm_tracking_was_successful(tracking))
    {
        append_tracking_target(data, tracking, recording_frame_no);
    }
    else
    {
        gm_message(data->log,
                   "Skipping frame %d (failed to track)",
                   recording_frame_no);
    }

    gm_tracking_unref(tracking);
}

static bool
make_target_sequence_has_time_step_elapsed(Data *data)
{
    /* Note that we may pass more frames than necessary to gm_context for
     * tracking due to the latency before ->last_written_timestamp is
     * updated, but for large data->time_steps we can still avoid a lot of
     * redundant tracking work by skipping unwanted frames at this point.
     */
    uint64_t elapsed = UINT64_MAX;
    if (data->make_targets.last_written_timestamp) {
        elapsed = (data->last_depth_frame->timestamp -
                   data->make_targets.last_written_timestamp);
    }

    return (elapsed >= data->time_step);
}

static void
make_target_sequence_end(Data *data)
{
    fclose(data->make_targets.index);
    data->make_targets.index = NULL;
}

static struct command {
    const char *name;
    const char *desc;

    /* Required:
     * Parse any command-specific commandline arguments.
     *
     * Must also set data->recording_dir, which should be parsed as a tool
     * argument, but expected by the common setup code.
     */
    bool (*argparse)(Data *data, int argc, char **argv);

    /* Optional:
     * Initialize any state before the mainloop starts
     */
    bool (*start)(Data *data);

    /* Optional:
     * Called whenever tracking results are ready (not necessarily
     * successfully tracked).
     */
    void (*on_tracking_ready)(Data *data);

    /* Optional:
     * To handle the -t,--time commandline option each tool needs
     * to be able to determine if more than data->time_step
     * nanoseconds have elapsed since the tool successfully
     * consumed a frame.
     *
     * This is called before passing a new frame to gm_context via
     * gm_context_notify_frame, but if time_step time has not yet
     * elapsed then the frame will be skipped. Some false positives
     * may be ok (they will result in redundtly tracking a frame) if
     * if simplifies the tool (the tool should also handle
     * data->time_step checks in its on_tracking_ready callback).
     *
     * If NULL, then true is assumed.
     */
    bool (*has_time_step_elapsed)(Data *data);

    /* Optional:
     * Clean up state
     */
    void (*end)(Data *data);
} commands[] = {
    {
        "make_target_sequence",
        "Build a sequence of target poses from tracked frames",
        make_target_sequence_argparse,
        make_target_sequence_start,
        make_target_sequence_tracking_ready,
        make_target_sequence_has_time_step_elapsed,
        make_target_sequence_end,
    },
};


static bool
check_complete(Data *data, int recording_frame_no)
{
    if (recording_frame_no >= data->recording_frame_prop->int_state.max ||
        (data->end_frame && (recording_frame_no >= data->end_frame)))
    {
        data->finished = true;
    }

    return data->finished;
}

static void
handle_device_frame_updates(Data *data)
{
    if (!data->device_frame_ready)
        return;

    int recording_frame_no = data->notified_frame_no;

    gm_debug(data->log, "Handling device _FRAME_READY (recording_frame_no=%d)",
             recording_frame_no);

    {
        /* NB: gm_device_get_latest_frame will give us a _ref() */
        struct gm_frame *device_frame = gm_device_get_latest_frame(data->device);
        if (!device_frame) {
            return;
        }

        if (device_frame->depth) {
            if (data->last_depth_frame) {
                gm_frame_unref(data->last_depth_frame);
            }
            data->last_depth_frame = gm_frame_ref(device_frame);
            data->last_depth_frame_no = recording_frame_no;
            gm_debug(data->log, "recording frame %d included depth buffer",
                     recording_frame_no);
        }

        if (device_frame->video) {
            if (data->last_video_frame) {
                gm_frame_unref(data->last_video_frame);
            }
            data->last_video_frame = gm_frame_ref(device_frame);
            data->last_video_frame_no = recording_frame_no;
            gm_debug(data->log, "recording frame %d included video buffer",
                     recording_frame_no);
        }

        gm_frame_unref(device_frame);
    }

    if (data->context_needs_frame &&
        data->last_video_frame && data->last_depth_frame)
    {
        if (data->last_video_frame != data->last_depth_frame) {
            struct gm_frame *full_frame =
                gm_device_combine_frames(data->device,
                                         data->last_depth_frame,
                                         data->last_depth_frame,
                                         data->last_video_frame);

            // We don't need the individual frames any more
            gm_frame_unref(data->last_depth_frame);
            gm_frame_unref(data->last_video_frame);

            data->last_depth_frame = full_frame;
            data->last_video_frame = gm_frame_ref(full_frame);
        }

        int end_frame = data->end_frame ? data->end_frame :
            data->recording_frame_prop->int_state.max;

        if (recording_frame_no >= data->begin_frame &&
            recording_frame_no < end_frame)
        {
            bool time_step_elapsed = true;
            if (commands[data->command_index].has_time_step_elapsed)
                time_step_elapsed = commands[data->command_index].has_time_step_elapsed(data);

            if (time_step_elapsed) {
                if (gm_context_notify_frame(data->ctx, data->last_depth_frame)) {
                    gm_debug(data->log, "Sent recording frame to context (depth=%d, video=%d)",
                             data->last_depth_frame_no,
                             data->last_video_frame_no);

                    data->context_needs_frame = false;
                    data->last_tracking_frame_depth_no = data->last_depth_frame_no;
                    data->last_tracking_frame_video_no = data->last_video_frame_no;
                    data->last_tracking_timestamp = data->last_depth_frame->timestamp;
                }
            } else {
                gm_debug(data->log, "Skipping recording frame %d (-t,--time step not elapsed)",
                         recording_frame_no);
            }

            // We don't want to send duplicate frames to tracking, so discard now
            gm_frame_unref(data->last_depth_frame);
            data->last_depth_frame = NULL;
        } else {
            gm_debug(data->log, "Skipping out-of-bounds recording frame %d (begin = %d, end = %d",
                     recording_frame_no, data->begin_frame, end_frame);

            /* It's possible that the data->frame_time for sub sampling the
             * recording could take us past the data->end_frame of the
             * recording so we can't only rely on
             * handle_context_tracking_updates() to check for completion after
             * writing out targets...
             */
            check_complete(data, recording_frame_no);
        }
    }

    data->device_frame_ready = false;
}

static void
handle_context_tracking_updates(Data *data)
{
    if (!data->tracking_ready)
        return;

    gm_debug(data->log, "Handling context _TRACKING_READY");

    data->tracking_ready = false;

    if (commands[data->command_index].on_tracking_ready) {
        commands[data->command_index].on_tracking_ready(data);
    }

    /* Note this check is done regardless of whether the last tracking
     * was successful
     */
    int recording_frame_no = data->last_tracking_frame_depth_no;
    check_complete(data, recording_frame_no);

    /* We synchronize requesting device frames and waiting for tracking to
     * complete considering that we don't currently have a way to pipeline the
     * acquisition of multiple frames that may be buffered waiting to be
     * processed and we depend on a global device 'frame' counter to track
     * which recording frame we are handling.
     *
     * Resetting these indicates that we are ready to request a new device
     * frame (which will have the side-effect of bumping the 'frame' counter).
     */
    data->last_tracking_timestamp = 0;
    data->last_tracking_frame_depth_no = -1;
    data->last_tracking_frame_video_no = -1;
}

static void
handle_device_ready(Data *data)
{
    gm_debug(data->log, "Device ready");
    int max_depth_pixels = gm_device_get_max_depth_pixels(data->device);
    gm_context_set_max_depth_pixels(data->ctx, max_depth_pixels);

    int max_video_pixels = gm_device_get_max_video_pixels(data->device);
    gm_context_set_max_video_pixels(data->ctx, max_video_pixels);

    struct gm_ui_properties *props = gm_device_get_ui_properties(data->device);
    gm_props_set_bool(props, "loop", false);

    /* Normally when we play back a recording in glimpse_viewer then we
     * would like to see the speed of motion / framerate match the original
     * capture speed/framerate. To achieve that then the IO code for
     * reading frames will skip over frames if it's not keeping up or throttle
     * frame delivery if going too fast.
     *
     * In this case though we simply want to process every frame we have in the
     * recording as quickly as possible, regardless of how long it takes to
     * process each frame so we disable any wall-clock time synchronization.
     */
    gm_props_set_bool(props, "frame_skip", false);
    gm_props_set_bool(props, "frame_throttle", false);

    data->recording_frame_prop = gm_props_lookup(props, "frame");
    if (data->begin_frame)
        gm_prop_set_int(data->recording_frame_prop, data->begin_frame);

    char *catch_err = NULL;
    const char *device_config = "glimpse-device.json";
    if (!gm_device_load_config_asset(data->device,
                                     device_config,
                                     &catch_err))
    {
        gm_warn(data->log, "Didn't open device config: %s", catch_err);
        free(catch_err);
        catch_err = NULL;
    }

    gm_device_start(data->device);
    gm_context_enable(data->ctx);
}

static void
handle_device_event(Data *data, struct gm_device_event *event)
{
    switch (event->type) {
    case GM_DEV_EVENT_READY:
        handle_device_ready(data);
        break;

    case GM_DEV_EVENT_FRAME_READY:
        /* To avoid redundant work; just in case there are multiple
         * _FRAME_READY notifications backed up then we squash them together
         * and handle after we've iterated all outstanding events...
         *
         * (See handle_device_frame_updates())
         */
        data->device_frame_ready = true;
        break;
    }

    gm_device_event_free(event);
}

static void
handle_context_event(Data *data, struct gm_event *event)
{
    switch (event->type) {
    case GM_EVENT_REQUEST_FRAME:
        gm_debug(data->log, "Received context _REQUEST_FRAME event");
        data->context_needs_frame = true;
        break;
    case GM_EVENT_TRACKING_READY:
        gm_debug(data->log, "Received context _TRACKING_READY event");
        /* To avoid redundant work; just in case there are multiple
         * _TRACKING_READY notifications backed up then we squash them together
         * and handle after we've iterated all outstanding events...
         *
         * (See handle_context_tracking_updates())
         */
        data->tracking_ready = true;
        break;
    }

    gm_context_event_free(event);
}

static void
event_loop_iteration(Data *data)
{
    gm_debug(data->log, "Processing events");

    pthread_mutex_lock(&data->event_queue_lock);
    std::swap(data->events_front, data->events_back);
    pthread_mutex_unlock(&data->event_queue_lock);

    for (unsigned i = 0; i < data->events_front->size(); i++) {
        struct event event = (*data->events_front)[i];

        switch (event.type) {
        case EVENT_DEVICE:
            handle_device_event(data, event.device_event);
            break;
        case EVENT_CONTEXT:
            handle_context_event(data, event.context_event);
            break;
        }
    }

    data->events_front->clear();

    /* To avoid redundant work; just in case there are multiple _TRACKING_READY
     * or _FRAME_READY notifications backed up then we squash them together and
     * handle after we've iterated all outstanding events...
     */
    handle_device_frame_updates(data);
    handle_context_tracking_updates(data);

    /* We synchronize requesting device frames and waiting for tracking to
     * complete considering that we don't currently have a way to pipeline the
     * acquisition of multiple frames that may be buffered waiting to be
     * processed and we depend on a global device 'frame' counter to track
     * which recording frame we are handling.
     */
    if (data->context_needs_frame &&
        data->last_tracking_timestamp == 0)
    {
        gm_debug(data->log, "requesting new DEPTH|VIDEO buffers");
        gm_device_request_frame(data->device, (GM_REQUEST_FRAME_DEPTH |
                                               GM_REQUEST_FRAME_VIDEO));
    }
}

/* XXX:
 *
 * It's undefined what thread an event notification is delivered on
 * and undefined what locks may be held by the device/context subsystem
 * (and so reentrancy may result in a dead-lock).
 *
 * Events should not be processed synchronously within notification callbacks
 * and instead work should be queued to run on a known thread with a
 * deterministic state for locks...
 */
static void
on_event_cb(struct gm_context *ctx,
            struct gm_event *context_event, void *user_data)
{
    Data *data = (Data *)user_data;

    gm_debug(data->log, "Received context event, type = %d", context_event->type);

    struct event event = {};
    event.type = EVENT_CONTEXT;
    event.context_event = context_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_cond_signal(&data->event_notify_cond);
    pthread_mutex_unlock(&data->event_queue_lock);
}

static void
on_device_event_cb(struct gm_device_event *device_event,
                   void *user_data)
{
    Data *data = (Data *)user_data;

    gm_debug(data->log, "Received device event, type = %d", device_event->type);

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);

    if (device_event->type == GM_DEV_EVENT_FRAME_READY) {
        /* XXX: Ideally the device frame would include a property/value that
         * let us know the recording frame number that it corresponds to but
         * for now we depend on reading the device global 'frame' property.
         *
         * XXX: It's quite hacky but we read the property now because this
         * callback is invoked by (and synchronized with) the recording IO
         * thread so we know we can safely read the value without racing
         * with the playback IO.
         */
        data->notified_frame_no = gm_prop_get_int(data->recording_frame_prop);
    }

    data->events_back->push_back(event);
    pthread_cond_signal(&data->event_notify_cond);
    pthread_mutex_unlock(&data->event_queue_lock);
}

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: recordings-tool [options...] <command> [command_options...]\n"
"\n"
"Runs various commands for processing a glimpse_viewer recording.\n"
"\n"
"  common options:\n"
"\n"
"    -b, --begin=NUMBER   Begin on n frame (default: 1)\n"
"    -e, --end=NUMBER     End on this frame (default: unset)\n"
"    -t, --time=NUMBER    Minimum number of seconds between frames (default: 0)\n"
"    -v, --verbose        Verbose output\n"
"    -h, --help           Display this help\n"
"\n"
"  commands:\n"
"\n"
);
    for (int i = 0; i < ARRAY_LEN(commands); i++) {
        fprintf(stream, "    %s - %s\n",
                commands[i].name,
                commands[i].desc);
    }
    fprintf(stream, "\n");
}

int
main(int argc, char **argv)
{
    bool verbose_output = false;

    // Leading + ensures parsing stops at first non-option (i.e.
    // sub-command name)...
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
            data.begin_frame = atoi(optarg);
            if (data.begin_frame < 0)
                data.begin_frame = 0;
            break;
        case 'e':
            data.end_frame = atoi(optarg);
            if (data.end_frame < 0)
                data.end_frame = 0;
            break;
        case 't':
            data.time_step = (uint64_t)(strtod(optarg, NULL) * 1000000000.0);
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

    if (optind == argc) {
        fprintf(stderr, "No command specified\n\n");
        print_usage(stderr);
        return 1;
    }

    data.log = gm_logger_new(NULL, (void *)&verbose_output);

    const char *command = argv[optind];
    data.command_index = -1;

    for (int i = 0; i < ARRAY_LEN(commands); i++) {
        if (strcmp(commands[i].name, command) == 0) {
            data.command_index = i;
            gm_assert(data.log, commands[i].argparse != NULL,
                      "%s tool missing argparse implementation",
                      commands[i].name);
            break;
        }
    }
    if (data.command_index == -1) {
        fprintf(stderr, "Unknown command %s\n\n", command);
        print_usage(stderr);
        return 1;
    }

    if (commands[data.command_index].argparse) {
        if (!commands[data.command_index].argparse(&data,
                                                   argc - optind,
                                                   &argv[optind]))
        {
            return 1;
        }
    }


    if (data.end_frame && data.end_frame < data.begin_frame) {
        fprintf(stderr, "End frame should be >= begin frame\n\n");
        return 1;
    }

    const char *assets_root_env = getenv("GLIMPSE_ASSETS_ROOT");
    char *assets_root = strdup(assets_root_env ? assets_root_env : "");
    gm_set_assets_root(data.log, assets_root);

    pthread_mutex_init(&data.event_queue_lock, NULL);
    pthread_cond_init(&data.event_notify_cond, NULL);
    data.events_front = new std::vector<struct event>();
    data.events_back = new std::vector<struct event>();

    gm_debug(data.log, "Creating context");
    data.ctx = gm_context_new(data.log, NULL);
    gm_context_set_event_callback(data.ctx, on_event_cb, &data);

    char *open_err = NULL;
    struct gm_asset *config_asset = gm_asset_open(data.log,
                                                  "glimpse-config.json",
                                                  GM_ASSET_MODE_BUFFER,
                                                  &open_err);
    if (config_asset) {
        const char *buf = (const char *)gm_asset_get_buffer(config_asset);
        JSON_Value *json_config = json_parse_string(buf);
        gm_context_set_config(data.ctx, json_config);
        json_value_free(json_config);
        gm_asset_close(config_asset);
    } else {
        gm_warn(data.log, "Failed to open glimpse-config.json: %s",
                open_err);
        free(open_err);
    }

    if (commands[data.command_index].start) {
        if (!commands[data.command_index].start(&data)) {
            return 1;
        }
    }

    struct gm_device_config config = {};
    config.type = GM_DEVICE_RECORDING;
    config.recording.path = data.recording_dir;

    /* This option ensures that only one recording frame will be read per
     * gm_device_request_frame call, which helps us be sure we can process all
     * the frames in a recording.
     */
    config.recording.lockstep_io = true;

    gm_debug(data.log, "Opening device");
    data.device = gm_device_open(data.log, &config, NULL);
    gm_device_set_event_callback(data.device, on_device_event_cb, &data);
    gm_debug(data.log, "Committing device config");
    gm_device_commit_config(data.device, NULL);


    gm_debug(data.log, "Main Loop...");
    while (!data.finished) {
        pthread_mutex_lock(&data.event_queue_lock);
        if (!data.events_back->size()) {
            pthread_cond_wait(&data.event_notify_cond,
                              &data.event_queue_lock);
        }
        pthread_mutex_unlock(&data.event_queue_lock);

        event_loop_iteration(&data);
    }

    gm_device_stop(data.device);

    for (unsigned i = 0; i < data.events_back->size(); i++) {
        struct event event = (*data.events_back)[i];

        switch (event.type) {
        case EVENT_DEVICE:
            gm_device_event_free(event.device_event);
            break;
        case EVENT_CONTEXT:
            gm_context_event_free(event.context_event);
            break;
        }
    }

    gm_context_destroy(data.ctx);

    if (data.last_depth_frame) {
        gm_frame_unref(data.last_depth_frame);
    }
    if (data.last_video_frame) {
        gm_frame_unref(data.last_video_frame);
    }

    gm_device_close(data.device);

    delete data.events_front;
    delete data.events_back;

    if (commands[data.command_index].end) {
        commands[data.command_index].end(&data);
    }

    gm_logger_destroy(data.log);

    return 0;
}
