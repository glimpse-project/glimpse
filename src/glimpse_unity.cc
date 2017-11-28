#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <IUnityInterface.h>
#include <IUnityGraphics.h>

#ifndef ANDROID
#include <epoxy/gl.h>
#endif

#include <vector>

#include "half.hpp"

#include "glimpse_log.h"
#include "glimpse_device.h"
#include "glimpse_context.h"

using half_float::half;

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

struct glimpse_data
{
    struct gm_logger *log;
    struct gm_context *ctx;
    struct gm_device *device;

    /* A convenience for accessing the depth_camera_intrinsics.width/height */
    int depth_width;
    int depth_height;

    /* A convenience for accessing the video_camera_intrinsics.width/height */
    int video_width;
    int video_height;

    /* When we request gm_device for a frame we set requirements for what the
     * frame should include. We track the requirements so we avoid sending
     * subsequent frame requests that would downgrade the requirements
     */
    uint64_t pending_frame_requirements;

    /* Set when gm_device sends a _FRAME_READY device event */
    bool device_frame_ready;

    /* Once we've been notified that there's a device frame ready for us then
     * we store the latest frame from gm_device_get_latest_frame() here...
     *
     * NB: this frame is only valid to access up until the next call to
     * gm_device_get_latest_frame() because the gm_device api is free to
     * recycle the back buffers that are part of a frame.
     */
    struct gm_frame *device_frame;

    /* Set when gm_context sends a _REQUEST_FRAME event */
    bool context_needs_frame;
    /* Set when gm_context sends a _TRACKING_READY event */
    bool tracking_ready;

    /* Once we've been notified that there's a skeleton tracking update for us
     * then we store the latest tracking data from
     * gm_context_get_latest_tracking() here...
     */
    struct gm_tracking *latest_tracking;

    /* Events from the gm_context and gm_device apis may be delivered via any
     * arbitrary thread which we don't want to block, and at a time where
     * the gm_ apis may not be reentrant due to locks held during event
     * notification
     */
    pthread_mutex_t event_queue_lock;
    std::vector<struct event> *events_back;
    std::vector<struct event> *events_front;
};

static void (*unity_log_function)(int level,
                                  const char *context,
                                  const char *msg);

static GLuint video_texture;
static int video_texture_width;
static int video_texture_height;

static float unity_current_time;
static IUnityInterfaces *unity_interfaces;
static IUnityGraphics *unity_graphics;

static UnityGfxRenderer unity_renderer_type = kUnityGfxRendererNull;

static struct glimpse_data *glimpse_data;


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_notify_log_function(void (*log_func)(int level,
                                              const char *context,
                                              const char *msg))
{
    unity_log_function = log_func;
}

static void
logger_cb(struct gm_logger *logger,
          enum gm_log_level level,
          const char *context,
          const char *backtrace,
          const char *format,
          va_list ap,
          void *user_data)
{
    char *msg = NULL;

    if (vasprintf(&msg, format, ap) > 0) {
        unity_log_function(level, context, msg);
        free(msg);
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_set_time(float time)
{
    unity_current_time = time;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_set_video_texture(void *handle_as_ptr, int width, int height)
{
    video_texture = *(GLuint *)handle_as_ptr;
    video_texture_width = width;
    video_texture_height = height;
}

static void UNITY_INTERFACE_API
on_graphics_device_event_cb(UnityGfxDeviceEventType type)
{
    switch (type) {
    case kUnityGfxDeviceEventInitialize:
        unity_renderer_type = unity_graphics->GetRenderer();
        break;
    case kUnityGfxDeviceEventShutdown:
        unity_renderer_type = kUnityGfxRendererNull;
        break;
    default:
        break;
    }
}

static void UNITY_INTERFACE_API
on_render_event_cb(int event)
{
    unity_log("Render Event %d DEBUG\n", event);
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_render_event_callback(void)
{
    return on_render_event_cb;
}

/* NB: it's undefined what thread this is called on and we are currently
 * assuming it's safe to call gm_device_request_frame() from any thread
 * considering that it just sets a bitmask and signals a condition variable.
 */
static void
on_event_cb(struct gm_context *ctx,
            struct gm_event *context_event, void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    unity_log("GLIMPSE: Context Event\n");

    struct event event = {};
    event.type = EVENT_CONTEXT;
    event.context_event = context_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
}

static void
on_device_event_cb(struct gm_device *dev,
                   struct gm_device_event *device_event,
                   void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    unity_log("GLIMPSE: Device Event\n");

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_init(void)
{
    glimpse_data->log = gm_logger_new(logger_cb, glimpse_data);

    struct gm_device_config config = {};
    config.type = GM_DEVICE_KINECT;
    glimpse_data->device = gm_device_open(glimpse_data->log, &config, NULL);

    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(glimpse_data->device);
    glimpse_data->depth_width = depth_intrinsics->width;
    glimpse_data->depth_height = depth_intrinsics->height;

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(glimpse_data->device);
    glimpse_data->video_width = video_intrinsics->width;
    glimpse_data->video_height = video_intrinsics->height;

    glimpse_data->ctx = gm_context_new(glimpse_data->log, NULL);

    gm_context_set_depth_camera_intrinsics(glimpse_data->ctx, depth_intrinsics);
    gm_context_set_video_camera_intrinsics(glimpse_data->ctx, video_intrinsics);

    /* NB: there's no guarantee about what thread these event callbacks
     * might be invoked from...
     */
    gm_context_set_event_callback(glimpse_data->ctx, on_event_cb, glimpse_data);
    gm_device_set_event_callback(glimpse_data->device, on_device_event_cb, glimpse_data);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_run(void)
{
    unity_log("GLIMPSE: Run\n");
    gm_device_start(glimpse_data->device);
    gm_context_enable(glimpse_data->ctx);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_stop(void)
{
    unity_log("GLIMPSE: Stop\n");
    gm_context_disable(glimpse_data->ctx);
    //gm_device_stop(glimpse_data->device);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_terminate(void)
{
    unity_log("GLIMPSE: Terminate\n");
    gm_context_destroy(glimpse_data->ctx);
    gm_device_close(glimpse_data->device);
}

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces *interfaces)
{
    unity_interfaces = interfaces;
    unity_graphics = interfaces->Get<IUnityGraphics>();
    unity_graphics->RegisterDeviceEventCallback(on_graphics_device_event_cb);

    on_graphics_device_event_cb(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload(void)
{
    unity_graphics->UnregisterDeviceEventCallback(on_graphics_device_event_cb);
}
