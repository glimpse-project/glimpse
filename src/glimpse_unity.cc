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

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <IUnityInterface.h>
#include <IUnityGraphics.h>

#include <epoxy/gl.h>

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "half.hpp"

#include "xalloc.h"
#include "glimpse_log.h"
#include "glimpse_device.h"
#include "glimpse_context.h"

#undef GM_LOG_CONTEXT

#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Plugin"
#include <android/log.h>
#include <jni.h>
#else
#define GM_LOG_CONTEXT "unity_plugin"
#endif

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
    FILE *log_fp;

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

    bool registered_gl_debug_callback;
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

static struct glimpse_data *plugin_data;


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
          struct gm_backtrace *backtrace,
          const char *format,
          va_list ap,
          void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;
    char *msg = NULL;

    if (vasprintf(&msg, format, ap) > 0) {
#ifdef __ANDROID__
        switch (level) {
        case GM_LOG_ASSERT:
            __android_log_print(ANDROID_LOG_FATAL, context, "%s", msg);
            break;
        case GM_LOG_ERROR:
            __android_log_print(ANDROID_LOG_ERROR, context, "%s", msg);
            break;
        case GM_LOG_WARN:
            __android_log_print(ANDROID_LOG_WARN, context, "%s", msg);
            break;
        case GM_LOG_INFO:
            __android_log_print(ANDROID_LOG_INFO, context, "%s", msg);
            break;
        case GM_LOG_DEBUG:
            __android_log_print(ANDROID_LOG_DEBUG, context, "%s", msg);
            break;
        }
#else
        if (unity_log_function)
            unity_log_function(level, context, msg);
#endif

        if (data->log_fp) {
            switch (level) {
            case GM_LOG_ERROR:
                fprintf(data->log_fp, "%s: ERROR: ", context);
                break;
            case GM_LOG_WARN:
                fprintf(data->log_fp, "%s: WARN: ", context);
                break;
            default:
                fprintf(data->log_fp, "%s: ", context);
            }

            fprintf(data->log_fp, "%s\n", msg);

            if (backtrace) {
                int line_len = 100;
                char *formatted = (char *)alloca(backtrace->n_frames * line_len);

                gm_logger_get_backtrace_strings(logger, backtrace,
                                                line_len, (char *)formatted);
                for (int i = 0; i < backtrace->n_frames; i++) {
                    char *line = formatted + line_len * i;
                    fprintf(data->log_fp, "> %s\n", line);
                }
            }
        }

        free(msg);
    }
}

static void
logger_abort_cb(struct gm_logger *logger,
                void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    if (data->log_fp) {
        fprintf(data->log_fp, "ABORT\n");
        fflush(data->log_fp);
        fclose(data->log_fp);
    }

    abort();
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_log(int level, const char *msg)
{
    if (plugin_data)
        gm_log(plugin_data->log, (enum gm_log_level)level, "GlimpseUnity", "%s", msg);
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

/* If we've already requested gm_device for a frame then this won't submit
 * a request that downgrades the requirements
 */
static void
request_device_frame(struct glimpse_data *data, uint64_t requirements)
{
    uint64_t new_requirements = data->pending_frame_requirements | requirements;

    if (data->pending_frame_requirements != new_requirements) {
        gm_device_request_frame(data->device, new_requirements);
        data->pending_frame_requirements = new_requirements;
    }
}

static void
handle_device_frame_updates(struct glimpse_data *data)
{
    //ProfileScopedSection(UpdatingDeviceFrame);
    //bool upload = false;

    if (!data->device_frame_ready)
        return;

    if (data->device_frame) {
        //ProfileScopedSection(FreeFrame);
        gm_frame_unref(data->device_frame);
    }

    {
        //ProfileScopedSection(GetLatestFrame);
        /* NB: gm_device_get_latest_frame will give us a _ref() */
        data->device_frame = gm_device_get_latest_frame(data->device);
        assert(data->device_frame);
        //upload = true;
    }

    if (data->context_needs_frame) {
        //ProfileScopedSection(FwdContextFrame);

        data->context_needs_frame =
            !gm_context_notify_frame(data->ctx, data->device_frame);
    }

    data->device_frame_ready = false;

    {
        //ProfileScopedSection(DeviceFrameRequest);

        /* immediately request a new frame since we want to render the camera
         * at the native capture rate, even though we might not be tracking
         * at that rate.
         *
         * Note: the requirements may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, GM_REQUEST_FRAME_VIDEO);
    }

#if 0
    if (upload) {
        ProfileScopedSection(UploadFrameTextures);

        /*
         * Update luminance from RGB camera
         */
        glBindTexture(GL_TEXTURE_2D, gl_lum_tex);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        void *video_front = gm_frame_get_video_buffer(data->device_frame);
        enum gm_format video_format = gm_frame_get_video_format(data->device_frame);

        assert(video_format == GM_FORMAT_LUMINANCE_U8);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                     data->video_width, data->video_height,
                     0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
    }
#endif
}

static void
handle_context_tracking_updates(struct glimpse_data *data)
{
    if (!data->tracking_ready)
        return;

    data->tracking_ready = false;

    struct gm_tracking *tracking = gm_context_get_latest_tracking(data->ctx);
    if (data->latest_tracking)
        gm_tracking_unref(data->latest_tracking);
    data->latest_tracking = tracking;

    assert(data->latest_tracking);

    //upload_tracking_textures(data);
}

static void
handle_device_event(struct glimpse_data *data, struct gm_device_event *event)
{
    switch (event->type) {
    case GM_DEV_EVENT_FRAME_READY:
        gm_debug(data->log, "GM_DEV_EVENT_FRAME_READY\n");

        /* It's always possible that we will see an event for a frame
         * that was ready before we upgraded the requirements for what
         * we need, so we skip notifications for frames we can't use.
         */
        if (event->frame_ready.met_requirements ==
            data->pending_frame_requirements)
        {
            data->pending_frame_requirements = 0;
            data->device_frame_ready = true;
        }
        break;
    }

    gm_device_event_free(event);
}

static void
handle_context_event(struct glimpse_data *data, struct gm_event *event)
{
    switch (event->type) {
    case GM_EVENT_REQUEST_FRAME:
        gm_debug(data->log, "GM_EVENT_REQUEST_FRAME\n");
        data->context_needs_frame = true;
        request_device_frame(data,
                             (GM_REQUEST_FRAME_DEPTH |
                              GM_REQUEST_FRAME_VIDEO));
        break;
    case GM_EVENT_TRACKING_READY:
        gm_debug(data->log, "GM_EVENT_TRACKING_READY\n");
        data->tracking_ready = true;
        break;
    }

    gm_context_event_free(event);
}

static void
process_events(struct glimpse_data *data)
{
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

    handle_device_frame_updates(data);
    handle_context_tracking_updates(data);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_process_events(void)
{
    process_events(plugin_data);
}

static void
on_khr_debug_message_cb(GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum gl_severity,
                        GLsizei length,
                        const GLchar *message,
                        void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    /* Ignore various noise generated by the Unity internals */
    if (gl_severity < GL_DEBUG_SEVERITY_MEDIUM ||
        source == GL_DEBUG_SOURCE_APPLICATION ||
        source == GL_DEBUG_SOURCE_THIRD_PARTY)
        return;

    switch (gl_severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        gm_log(data->log, GM_LOG_ERROR, "Glimpse GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        gm_log(data->log, GM_LOG_WARN, "Glimpse GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        gm_log(data->log, GM_LOG_WARN, "Glimpse GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        gm_log(data->log, GM_LOG_INFO, "Glimpse GL", "%s", message);
        break;
    }
}

static void UNITY_INTERFACE_API
on_render_event_cb(int event)
{
    gm_debug(plugin_data->log, "Render Event %d DEBUG\n", event);

    /* We just assume Unity isn't registering a GL debug callback and
     * cross our fingers...
     */
    if (!plugin_data->registered_gl_debug_callback) {
        glDebugMessageControl(GL_DONT_CARE, /* source */
                              GL_DONT_CARE, /* type */
                              GL_DONT_CARE, /* severity */
                              0,
                              NULL,
                              false);

        glDebugMessageControl(GL_DONT_CARE, /* source */
                              GL_DEBUG_TYPE_ERROR,
                              GL_DONT_CARE, /* severity */
                              0,
                              NULL,
                              true);

        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback((GLDEBUGPROC)on_khr_debug_message_cb, plugin_data);
        plugin_data->registered_gl_debug_callback = true;
    }

    gm_context_render_thread_hook(plugin_data->ctx);
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_render_event_callback(void)
{
    return on_render_event_cb;
}

/* NB: it's undefined what thread this is called on so we queue events to
 * be processed by gm_unity_process_events() during the GlimpseRuntime
 * script's Update().
 */
static void
on_event_cb(struct gm_context *ctx,
            struct gm_event *context_event, void *user_data)
{
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    gm_debug(data->log, "GLIMPSE: Context Event\n");

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

    gm_debug(data->log, "GLIMPSE: Device Event\n");

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_init(void)
{
    struct glimpse_data *data = new glimpse_data();

    plugin_data = data;

    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

    gm_debug(data->log, "GLIMPSE: Init\n");

    switch (unity_renderer_type) {
    case kUnityGfxRendererOpenGLES20:
        gm_debug(data->log, "OpenGL ES 2.0 Renderer");
        break;
    case kUnityGfxRendererOpenGLES30:
        gm_debug(data->log, "OpenGL ES 3.0 Renderer");
        break;
    case kUnityGfxRendererOpenGLCore:
        gm_debug(data->log, "OpenGL Core Renderer");
        break;
    default:
        gm_debug(data->log, "Unexpected Unity Renderer %d",
                 (int)unity_renderer_type);
        break;
    }

#ifdef __ANDROID__
    // During development on Android we are manually uploading recording and
    // training models to /sdcard on test devices so that build+upload cycles
    // of packages built via Unity can be as quick as possible by reducing
    // the size of .apk we have to repeatedly upload.
    //
#define ANDROID_ASSETS_ROOT "/sdcard/GlimpseUnity"
    setenv("GLIMPSE_ASSETS_ROOT", ANDROID_ASSETS_ROOT, true);
    setenv("FAKENECT_PATH", ANDROID_ASSETS_ROOT "/FakeRecording", true);
    data->log_fp = fopen(ANDROID_ASSETS_ROOT "/glimpse.log", "w");
#else
    data->log_fp = fopen("glimpse.log", "w");
#endif

    data->events_front = new std::vector<struct event>();
    data->events_back = new std::vector<struct event>();

    struct gm_device_config config = {};
    config.type = GM_DEVICE_KINECT;
    data->device = gm_device_open(data->log, &config, NULL);

    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(data->device);
    data->depth_width = depth_intrinsics->width;
    data->depth_height = depth_intrinsics->height;

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(data->device);
    data->video_width = video_intrinsics->width;
    data->video_height = video_intrinsics->height;

    data->ctx = gm_context_new(data->log, NULL);

    gm_context_set_depth_camera_intrinsics(data->ctx, depth_intrinsics);
    gm_context_set_video_camera_intrinsics(data->ctx, video_intrinsics);

    /* NB: there's no guarantee about what thread these event callbacks
     * might be invoked from...
     */
    gm_context_set_event_callback(data->ctx, on_event_cb, plugin_data);
    gm_device_set_event_callback(data->device, on_device_event_cb, plugin_data);

    return (intptr_t )data;
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_latest_tracking(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    gm_debug(data->log, "GLIMPSE: Get Latest Tracking");
    return (intptr_t)gm_context_get_latest_tracking(data->ctx);
}

extern "C" const float * UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_get_joint_positions(intptr_t plugin_handle,
                                      intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;
    int n_joints;

    gm_debug(data->log, "GLIMPSE: Tracking: Get Label Probabilities");

    const float *joints =
        gm_tracking_get_joint_positions(tracking, &n_joints);

    return joints;
}


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_unref(intptr_t plugin_handle, intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;

    gm_debug(data->log, "GLIMPSE: Tracking Unref %p (ref = %d)",
             tracking,
             tracking->ref);
    gm_tracking_unref(tracking);
}

static glm::mat4
intrinsics_to_project_matrix(struct gm_intrinsics *intrinsics,
                             float near, float far)
{
  float width = intrinsics->width;
  float height = intrinsics->height;

  float scalex = near / intrinsics->fx;
  float scaley = near / intrinsics->fy;

  float offsetx = (intrinsics->cx - width / 2.0) * scalex;
  float offsety = (intrinsics->cy - height / 2.0) * scaley;

  return glm::frustum(scalex * -width / 2.0f - offsetx,
                      scalex * width / 2.0f - offsetx,
                      scaley * height / 2.0f - offsety,
                      scaley * -height / 2.0f - offsety, near, far);
}

extern "C" const void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_video_projection(intptr_t plugin_handle, float *out_mat4)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    struct gm_intrinsics *intrinsics =
      gm_device_get_video_intrinsics(data->device);

    memcpy(out_mat4,
           glm::value_ptr(intrinsics_to_project_matrix(intrinsics, 0.5, 5)),
           sizeof(float) * 16);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_run(void)
{
    struct glimpse_data *data = plugin_data;

    gm_debug(data->log, "GLIMPSE: Run\n");
    gm_device_start(data->device);
    gm_context_enable(data->ctx);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_stop(void)
{
    struct glimpse_data *data = plugin_data;

    gm_debug(data->log, "GLIMPSE: Stop\n");
    gm_context_disable(data->ctx);
    //gm_device_stop(data->device);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_terminate(void)
{
    struct glimpse_data *data = plugin_data;

    gm_debug(data->log, "GLIMPSE: Terminate\n");

    /* Destroying the context' tracking pool will assert that all tracking
     * resources have been released first...
     */
    if (data->latest_tracking)
        gm_tracking_unref(data->latest_tracking);

    /* NB: It's our responsibility to be sure that there can be no asynchonous
     * calls into the gm_context api before we start to destroy it!
     *
     * We stop the device first because device callbacks result in calls
     * through to the gm_context api.
     *
     * We don't destroy the device first because destroying the context will
     * release device resources (which need to be release before the device
     * can be cleanly closed).
     */
    gm_device_stop(data->device);

    for (unsigned i = 0; i < data->events_back->size(); i++) {
        struct event event = (*data->events_back)[i];

        switch (event.type) {
        case EVENT_DEVICE:
            gm_device_event_free(event.device_event);
            break;
        case EVENT_CONTEXT:
            gm_context_event_free(event.context_event);
            break;
        }
    }

    gm_context_destroy(data->ctx);

    if (data->device_frame)
        gm_frame_unref(data->device_frame);

    gm_device_close(data->device);

    gm_debug(data->log, "Destroying logger");
    gm_logger_destroy(data->log);
    fclose(data->log_fp);
    unity_log_function = NULL;

    delete plugin_data;
    plugin_data = NULL;
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

#ifdef __ANDROID__
static jobject
instantiate_glimpse_test_class(JNIEnv* jni_env)
{
    jclass cls_JavaClass = jni_env->FindClass("com/impossible/glimpse/GlimpseTest");
    jmethodID mid_JavaClass = jni_env->GetMethodID(cls_JavaClass, "<init>", "()V");
    jobject obj_JavaClass = jni_env->NewObject(cls_JavaClass, mid_JavaClass);
    return jni_env->NewGlobalRef(obj_JavaClass);
}

extern "C" jint UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *jni_env = 0;
    vm->AttachCurrentThread(&jni_env, 0);

    instantiate_glimpse_test_class(jni_env);
    return JNI_VERSION_1_6;
}
#endif
