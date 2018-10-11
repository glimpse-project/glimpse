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

#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <unistd.h>

#include <IUnityInterface.h>
#include <IUnityGraphics.h>
#include <IUnityRenderingExtensions.h>

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
#include "glimpse_gl.h"
#include "glimpse_assets.h"
#include "glimpse_target.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#else
#define TARGET_OS_MAC 0
#define TARGET_OS_IOS 0
#define TARGET_OS_OSX 0
#endif

#if TARGET_OS_IOS == 1
#include "ios_utils.h"
#endif

#undef GM_LOG_CONTEXT

#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Plugin"
#include <android/log.h>
#include <jni.h>
#else
#define GM_LOG_CONTEXT "unity_plugin"
#endif

#if defined(__ANDROID__) || TARGET_OS_IOS == 1
#define GLSL_SHADER_VERSION "#version 300 es\n"
#define USE_GLES 1
#else
#define GLSL_SHADER_VERSION "#version 150\n"
#define USE_CORE_GL 1
#endif

#ifdef USE_TANGO
#include <tango_client_api.h>
#include <tango_support_api.h>
#endif

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)

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
    JSON_Value *config_val;
    JSON_Object *config;

    FILE *log_fp;
    struct gm_logger *log;

    int render_event_id;

    struct gm_context *ctx;
    struct gm_device *device;
    enum gm_device_type device_type;

    bool device_ready;

    /* When we request gm_device for a frame we set a buffers_mask for what the
     * frame should include. We track the buffers_mask so we avoid sending
     * subsequent frame requests that would downgrade the buffers_mask
     */
    uint64_t pending_frame_buffers_mask;

    /* Set when gm_device sends a _FRAME_READY device event */
    bool device_frame_ready;

    /* Once we've been notified that there's a device frame ready for us then
     * we store the latest frames from gm_device_get_latest_frame() here...
     *
     * We have the lock considering that unity's rendering happens in a separate
     * thread and also needs to access these pointers.
     */
    pthread_mutex_t swap_frames_lock;
    struct gm_frame *last_depth_frame;
    struct gm_frame *last_video_frame;

    /* When we come to render the background we take a reference on the latest
     * gm_frame before uploading to a texture for display.
     */
    struct gm_frame *visible_frame;

    /* When we pass the video frame to Unity, we take a reference on the latest
     * gm_frame.
     */
    struct gm_frame *texture_frame;

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
    pthread_cond_t event_notify_cond;
    std::vector<struct event> *events_back;
    std::vector<struct event> *events_front;

    bool registered_gl_debug_callback;

    GLuint video_program;

    GLuint attrib_quad_bo;

    /* Even though glEnable/DisableVertexAttribArray take unsigned integers,
     * these are signed because GL's glGetAttribLocation api returns attribute
     * locations as signed values where -1 means the attribute isn't
     * active. ...!?
     */
    GLint attrib_quad_pos;
    GLint attrib_quad_tex_coords;

    GLuint ar_video_tex_sampler;
    std::vector<GLuint> ar_video_queue;
    int ar_video_queue_len;
    int ar_video_queue_pos;
};

#ifdef __ANDROID__
static JavaVM *android_jvm_singleton;
#endif

static void (*unity_log_function)(int level,
                                  const char *context,
                                  const char *msg);

static IUnityInterfaces *unity_interfaces;
static IUnityGraphics *unity_graphics;

static UnityGfxRenderer unity_renderer_type = kUnityGfxRendererNull;

/* Considering that Unity renders in a separate thread which is awkward to
 * synchronize with from C# we need to take extra care while terminating
 * our plugin state because it's possible we might still see render
 * event callbacks which shouldn't try and access the plugin state
 */
static pthread_mutex_t life_cycle_lock = PTHREAD_MUTEX_INITIALIZER;
static bool terminating; // checked in on_render_event_cb()

// The render events can only be passed an event ID not a plugin_handle
// so we have to iterate all the current plugin handles to match
// with an event ID that's used instead of a pointer...
static pthread_mutex_t plugin_data_lock = PTHREAD_MUTEX_INITIALIZER;
static std::vector<struct glimpse_data *> all_plugin_data;

static int next_unique_event_id = 1;

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
#elif TARGET_OS_IOS == 1
        ios_log(msg);
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
            fflush(data->log_fp);
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
gm_unity_log(intptr_t plugin_handle, int level, const char *msg)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }

    gm_log(data->log, (enum gm_log_level)level, "GlimpseUnity", "%s", msg);
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
 * a request that downgrades the buffers_mask
 */
static void
request_device_frame(struct glimpse_data *data, uint64_t buffers_mask)
{
    uint64_t new_buffers_mask = data->pending_frame_buffers_mask | buffers_mask;

    if (data->pending_frame_buffers_mask != new_buffers_mask) {
        gm_device_request_frame(data->device, new_buffers_mask);
        data->pending_frame_buffers_mask = new_buffers_mask;
    }
}

static void
handle_device_frame_updates(struct glimpse_data *data)
{
    //ProfileScopedSection(UpdatingDeviceFrame);
    //bool upload = false;

    if (!data->device_frame_ready)
        return;

    {
        //ProfileScopedSection(GetLatestFrame);
        /* NB: gm_device_get_latest_frame will give us a _ref() */
        struct gm_frame *device_frame = gm_device_get_latest_frame(data->device);

        assert(device_frame);

        /* XXX: we have to consider that the rendering is in another thread
         * and we don't want to unref (and potentially free) the last frame
         * while the render thread might reference the same pointer...
         */
        pthread_mutex_lock(&data->swap_frames_lock);

        if (device_frame->depth) {
            if (data->last_depth_frame) {
                gm_frame_add_breadcrumb(data->last_depth_frame, "unity: discard old depth frame");
                gm_frame_unref(data->last_depth_frame);
            }
            gm_frame_ref(device_frame);
            data->last_depth_frame = device_frame;
            data->pending_frame_buffers_mask &= ~GM_REQUEST_FRAME_DEPTH;
            gm_frame_add_breadcrumb(device_frame, "unity: latest depth frame");
        }

        if (device_frame->video) {
            if (data->last_video_frame) {
                gm_frame_add_breadcrumb(data->last_video_frame, "unity: discard old video frame");
                gm_frame_unref(data->last_video_frame);
            }
            gm_frame_ref(device_frame);
            data->last_video_frame = device_frame;
            data->pending_frame_buffers_mask &= ~GM_REQUEST_FRAME_VIDEO;
            gm_frame_add_breadcrumb(device_frame, "unity: latest video frame");
            //upload = true;
        }

        pthread_mutex_unlock(&data->swap_frames_lock);

        gm_frame_unref(device_frame);
    }

    if (data->context_needs_frame &&
        data->last_depth_frame && data->last_video_frame) {
        //ProfileScopedSection(FwdContextFrame);

        // Combine the two video/depth frames into a single frame for gm_context
        if (data->last_depth_frame != data->last_video_frame) {
            struct gm_frame *full_frame =
                gm_device_combine_frames(data->device,
                                         data->last_depth_frame,
                                         data->last_depth_frame,
                                         data->last_video_frame);

            /* XXX: we have to consider that the rendering is in another thread
             * and we don't want to unref (and potentially free) the last frame
             * while the render thread might reference the same pointer...
             *
             * XXX: be careful making changes here to not end up holding this
             * lock for too long and blocking the rendering thread. E.g.
             * we avoid having the scope of the lock include the call into
             * gm_context_notify_frame().
             */
            pthread_mutex_lock(&data->swap_frames_lock);

            // We don't need the individual frames any more
            gm_frame_unref(data->last_depth_frame);
            gm_frame_unref(data->last_video_frame);

            data->last_depth_frame = full_frame;
            data->last_video_frame = gm_frame_ref(full_frame);

            pthread_mutex_unlock(&data->swap_frames_lock);
        }

        data->context_needs_frame =
            !gm_context_notify_frame(data->ctx, data->last_depth_frame);

        // We don't want to send duplicate frames to tracking, so discard now
        pthread_mutex_lock(&data->swap_frames_lock);
        gm_frame_unref(data->last_depth_frame);
        data->last_depth_frame = NULL;
        pthread_mutex_unlock(&data->swap_frames_lock);
    }

    data->device_frame_ready = false;

    {
        //ProfileScopedSection(DeviceFrameRequest);

        /* immediately request a new frame since we want to render the camera
         * at the native capture rate, even though we might not be tracking
         * at that rate.
         *
         * Note: the buffers_mask may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, GM_REQUEST_FRAME_VIDEO);
    }
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
handle_device_ready(struct glimpse_data *data)
{
    int max_depth_pixels =
        gm_device_get_max_depth_pixels(data->device);
    gm_context_set_max_depth_pixels(data->ctx, max_depth_pixels);

    int max_video_pixels =
        gm_device_get_max_video_pixels(data->device);
    gm_context_set_max_video_pixels(data->ctx, max_video_pixels);

    /*gm_context_set_depth_to_video_camera_extrinsics(data->ctx,
      gm_device_get_depth_to_video_extrinsics(data->device));*/

    char *catch_err = NULL;
    const char *device_config = json_object_get_string(data->config, "deviceConfig");
    if (device_config == NULL || strlen(device_config) == 0)
        device_config = "glimpse-device.json";
    if (!gm_device_load_config_asset(data->device,
                                     device_config,
                                     &catch_err))
    {
        gm_warn(data->log, "Didn't open device config: %s", catch_err);
        free(catch_err);
        catch_err = NULL;
    }

    data->device_ready = true;
}

static void
handle_device_event(struct glimpse_data *data, struct gm_device_event *event)
{
    switch (event->type) {
    case GM_DEV_EVENT_READY:
        handle_device_ready(data);
        break;
    case GM_DEV_EVENT_FRAME_READY:
        gm_debug(data->log, "GM_DEV_EVENT_FRAME_READY\n");

        /* NB: It's always possible that we will see an event for a frame
         * that was ready before we upgraded the buffers_mask for what
         * we need, so we skip notifications for frames we can't use.
         */
        if (event->frame_ready.buffers_mask & data->pending_frame_buffers_mask)
        {
            /* To avoid redundant work; just in case there are multiple
             * _FRAME_READY notifications backed up then we squash them
             * together and handle after we've iterated all outstanding
             * events...
             *
             * (See handle_device_frame_updates())
             */
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
event_loop_iteration(struct glimpse_data *data)
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

    /* To avoid redundant work; just in case there are multiple _TRACKING_READY
     * or _FRAME_READY notifications backed up then we squash them together and
     * handle after we've iterated all outstanding events...
     */
    handle_device_frame_updates(data);
    handle_context_tracking_updates(data);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_process_events(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }

    event_loop_iteration(data);
}

#if defined(USE_GLES) || defined(USE_CORE_GL)

static GLuint
gen_ar_video_texture(struct glimpse_data *data)
{
    GLuint ar_video_tex;

    glGenTextures(1, &ar_video_tex);

    GLenum target = GL_TEXTURE_2D;
    if (data->device_type == GM_DEVICE_TANGO) {
        target = GL_TEXTURE_EXTERNAL_OES;
    }

    glBindTexture(target, ar_video_tex);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return ar_video_tex;
}

static void
update_ar_video_queue_len(struct glimpse_data *data, int len)
{
    if (len >= data->ar_video_queue_len) {
        data->ar_video_queue_len = len;
        return;
    }
    glDeleteTextures(data->ar_video_queue.size(),
                     data->ar_video_queue.data());
    data->ar_video_queue.resize(0);
    data->ar_video_queue_len = len;
    data->ar_video_queue_pos = -1;
}

static GLuint
get_next_ar_video_tex(struct glimpse_data *data)
{
    if (data->ar_video_queue_len < 1) {
        update_ar_video_queue_len(data, 6);
    }

    if (data->ar_video_queue.size() < data->ar_video_queue_len) {
        GLuint ar_video_tex = gen_ar_video_texture(data);

        data->ar_video_queue_pos = data->ar_video_queue.size();
        data->ar_video_queue.push_back(ar_video_tex);
        return data->ar_video_queue.back();
    } else {
        data->ar_video_queue_pos =
            (data->ar_video_queue_pos + 1) % data->ar_video_queue_len;
        return data->ar_video_queue[data->ar_video_queue_pos];
    }
}

static GLuint
get_oldest_ar_video_tex(struct glimpse_data *data)
{
    if (data->ar_video_queue.size() < data->ar_video_queue_len) {
        return data->ar_video_queue[0];
    } else {
        int oldest = (data->ar_video_queue_pos + 1) % data->ar_video_queue_len;
        return data->ar_video_queue[oldest];
    }
}

static void
draw_video(struct glimpse_data *data,
           float *xyst_verts,
           int n_verts)
{
    /* Try our best to save and restore and GL state we use, to avoid confusing
     * any caching of state that Unity does.
     */
    GLint saved_program;
    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);

    GLint saved_array_buffer;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &saved_array_buffer);

    GLboolean saved_depth_test_enable = glIsEnabled(GL_DEPTH_TEST);

    if (!data->attrib_quad_bo) {
        glGenBuffers(1, &data->attrib_quad_bo);
    }
    /* XXX: we could just cache buffers for each rotation */
    glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_bo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n_verts,
                 xyst_verts, GL_STATIC_DRAW);

    if (!data->video_program) {
        const char *vert_shader =
            GLSL_SHADER_VERSION
            "precision mediump float;\n"
            "precision mediump int;\n"
            "in vec2 pos;\n"
            "in vec2 tex_coords_in;\n"
            "out vec2 tex_coords;\n"
            "void main() {\n"
            "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
            "  tex_coords = tex_coords_in;\n"
            "}\n";
        const char *frag_shader =
            GLSL_SHADER_VERSION
            "precision highp float;\n"
            "precision highp int;\n"
            "uniform sampler2D tex_sampler;\n"
            "in vec2 tex_coords;\n"
            "out lowp vec4 frag_color;\n"
            "void main() {\n"
            "  frag_color = texture(tex_sampler, tex_coords);\n"
            "}\n";
        const char *external_tex_frag_shader =
            GLSL_SHADER_VERSION
            "#extension GL_OES_EGL_image_external_essl3 : require\n"
            "precision highp float;\n"
            "precision highp int;\n"
            "uniform samplerExternalOES tex_sampler;\n"
            "in vec2 tex_coords;\n"
            "out lowp vec4 frag_color;\n"
            "void main() {\n"
            "  frag_color = texture(tex_sampler, tex_coords);\n"
            "}\n";


        if (data->device_type == GM_DEVICE_TANGO) {
            data->video_program = gm_gl_create_program(data->log,
                                                       vert_shader,
                                                       external_tex_frag_shader,
                                                       NULL);
        } else {
            data->video_program = gm_gl_create_program(data->log,
                                                       vert_shader,
                                                       frag_shader,
                                                       NULL);
        }

        data->attrib_quad_pos =
            glGetAttribLocation(data->video_program, "pos");
        data->attrib_quad_tex_coords =
            glGetAttribLocation(data->video_program, "tex_coords_in");

        data->ar_video_tex_sampler = glGetUniformLocation(data->video_program, "tex_sampler");
        glUseProgram(data->video_program);
        glUniform1i(data->ar_video_tex_sampler, 0);
    }

    glUseProgram(data->video_program);
    glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_bo);

    glEnableVertexAttribArray(data->attrib_quad_pos);
    glVertexAttribPointer(data->attrib_quad_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)0);

    if (data->attrib_quad_tex_coords != -1) {
        glEnableVertexAttribArray(data->attrib_quad_tex_coords);
        glVertexAttribPointer(data->attrib_quad_tex_coords,
                              2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)8);
    }

    GLuint ar_video_tex = get_oldest_ar_video_tex(data);
    GLenum target = GL_TEXTURE_2D;

#if defined(USE_CORE_GL) && defined(__linux__)
    glBindTextureUnit(0, ar_video_tex);
#else
#ifdef USE_TANGO
    if (data->device_type == GM_DEVICE_TANGO)
        target = GL_TEXTURE_EXTERNAL_OES;
#endif
    glBindTexture(target, ar_video_tex);
#endif

    /* Don't touch the depth buffer, otherwise everything rendered by later
     * cameras will likely be discarded...
     */
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLE_FAN, 0, n_verts);
    gm_debug(data->log, "draw_video");
    glDepthMask(GL_TRUE);

    glBindTexture(target, 0);

    glDisableVertexAttribArray(data->attrib_quad_pos);
    if (data->attrib_quad_tex_coords != -1)
        glDisableVertexAttribArray(data->attrib_quad_tex_coords);

    glBindBuffer(GL_ARRAY_BUFFER, saved_array_buffer);
    glUseProgram(saved_program);

    if (saved_depth_test_enable)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
}

static void
render_ar_video_background(struct glimpse_data *data)
{
    static int call_count = 0;
    int local_call_count = call_count++;
    struct gm_frame *new_frame = NULL;
    GLuint ar_video_tex = 0;

    gm_assert(data->log, !!data->ctx, "render_ar_video_background, NULL ctx");

    if (data->last_video_frame != data->visible_frame) {
        gm_info(data->log, "XXX %d: YES, last_video_frame != visible_frame (%p, %p)",
                local_call_count,
                data->last_video_frame,
                data->visible_frame);

        pthread_mutex_lock(&data->swap_frames_lock);

        new_frame = gm_frame_ref(data->last_video_frame);
        gm_info(data->log, "XXX %d: > new_frame = %p",
                local_call_count,
                new_frame);
        gm_frame_add_breadcrumb(new_frame, "render thread visible");

        if (data->visible_frame) {
            gm_frame_add_breadcrumb(data->visible_frame, "render thread discard");
            gm_frame_unref(data->visible_frame);
        }
        data->visible_frame = new_frame;

        pthread_mutex_unlock(&data->swap_frames_lock);
    } else {
        gm_info(data->log, "XXX %d: data->last_video_frame == data->visible_frame (%p, %p)",
                local_call_count,
                data->last_video_frame,
                data->visible_frame);
    }

    if (new_frame) {
        gm_info(data->log, "XXX %d: YES, new_frame", local_call_count);
        if (data->ar_video_queue_len == 0)
            update_ar_video_queue_len(data, 6); // XXX should be configurable

        if (data->device_type != GM_DEVICE_TANGO) {
                const struct gm_intrinsics *video_intrinsics =
                    &new_frame->video_intrinsics;
                int video_width = video_intrinsics->width;
                int video_height = video_intrinsics->height;

                ar_video_tex = get_next_ar_video_tex(data);
                glBindTexture(GL_TEXTURE_2D, ar_video_tex);

                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

                void *video_front = new_frame->video->data;
                enum gm_format video_format = new_frame->video_format;

                switch (video_format) {
                case GM_FORMAT_LUMINANCE_U8:
                    gm_debug(data->log, "uploading U8 %dx%d", video_width, video_height);
#ifdef USE_GLES
                    /* Annoyingly it doesn't seem to work on GLES3 + Android to
                     * upload a GL_RED texture + set a swizzle like we need to do
                     * for core GL (even though it looks like component swizzling
                     * is part of GLES3)
                     */
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                                 video_width, video_height,
                                 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
#else
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                                 video_width, video_height,
                                 0, GL_RED, GL_UNSIGNED_BYTE, video_front);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
#endif
                    break;

                case GM_FORMAT_RGB_U8:
                    gm_debug(data->log, "uploading RGB8 %dx%d", video_width, video_height);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                                 video_width, video_height,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, video_front);
                    break;
                case GM_FORMAT_BGR_U8:
                    gm_debug(data->log, "uploading BGR8 %dx%d", video_width, video_height);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_BGR,
                                 video_width, video_height,
                                 0, GL_BGR, GL_UNSIGNED_BYTE, video_front);
                    break;

                case GM_FORMAT_RGBX_U8:
                case GM_FORMAT_RGBA_U8:
                    gm_debug(data->log, "uploading RGBA8 %dx%d", video_width, video_height);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                 video_width, video_height,
                                 0, GL_RGBA, GL_UNSIGNED_BYTE, video_front);
                    break;
                case GM_FORMAT_BGRX_U8:
                case GM_FORMAT_BGRA_U8:
                    gm_debug(data->log, "uploading BGRAA8 %dx%d", video_width, video_height);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                 video_width, video_height,
                                 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, video_front);
                    break;

                case GM_FORMAT_UNKNOWN:
                case GM_FORMAT_Z_U16_MM:
                case GM_FORMAT_Z_F32_M:
                case GM_FORMAT_Z_F16_M:
                case GM_FORMAT_POINTS_XYZC_F32_M:
                    gm_assert(data->log, 0, "Unexpected format for video buffer");
                    break;
                }

                glBindTexture(GL_TEXTURE_2D, 0);
        } else {
#ifdef USE_TANGO
            ar_video_tex = get_next_ar_video_tex(data);
            if (TangoService_updateTextureExternalOes(
                    TANGO_CAMERA_COLOR, ar_video_tex,
                    NULL /* ignore timestamp */) != TANGO_SUCCESS)
            {
                gm_warn(data->log, "Failed to get a color image.");
            }
#endif
        }
    } else {
        gm_info(data->log, "XXX %d: !new_frame %p", local_call_count, new_frame);
    }

    if (data->last_video_frame && data->ar_video_queue.size()) {

        gm_info(data->log, "XXX %d: YES, (data->last_video_frame && data->ar_video_queue.size())",
                local_call_count);

        /* XXX: might not technically be the appropriate rotation but the
         * minimal buffer should mean it's not a big problem...
         */
        enum gm_rotation rotation = data->last_video_frame->camera_rotation;

        struct {
            float x, y, s, t;
        } xyst_verts[4] = {
            { -1,  1, 0, 0, }, //  0 -- 1
            {  1,  1, 1, 0, }, //  | \  |
            {  1, -1, 1, 1  }, //  |  \ |
            { -1, -1, 0, 1, }, //  3 -- 2
        };
        gm_debug(data->log, "rendering background with camera rotation of %d degrees",
                 ((int)rotation) * 90);

        switch (rotation) {
        case GM_ROTATION_0:
            break;
        case GM_ROTATION_90:
            xyst_verts[0].s = 1; xyst_verts[0].t = 0;
            xyst_verts[1].s = 1; xyst_verts[1].t = 1;
            xyst_verts[2].s = 0; xyst_verts[2].t = 1;
            xyst_verts[3].s = 0; xyst_verts[3].t = 0;
            break;
        case GM_ROTATION_180:
            xyst_verts[0].s = 1; xyst_verts[0].t = 1;
            xyst_verts[1].s = 0; xyst_verts[1].t = 1;
            xyst_verts[2].s = 0; xyst_verts[2].t = 0;
            xyst_verts[3].s = 1; xyst_verts[3].t = 0;
            break;
        case GM_ROTATION_270:
            xyst_verts[0].s = 0; xyst_verts[0].t = 1;
            xyst_verts[1].s = 0; xyst_verts[1].t = 0;
            xyst_verts[2].s = 1; xyst_verts[2].t = 0;
            xyst_verts[3].s = 1; xyst_verts[3].t = 1;
            break;
        }

        gm_debug(data->log, "UVs: %f,%f %f,%f %f,%f, %f,%f",
                 xyst_verts[0].s,
                 xyst_verts[0].t,
                 xyst_verts[1].s,
                 xyst_verts[1].t,
                 xyst_verts[2].s,
                 xyst_verts[2].t,
                 xyst_verts[3].s,
                 xyst_verts[3].t);

        draw_video(data, (float *)xyst_verts, 4);
    } else {
        gm_info(data->log, "XXX %d: !(data->last_video_frame && data->ar_video_queue.size())",
                local_call_count);
    }
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
#endif // GLES or CORE_GL

static void UNITY_INTERFACE_API
on_render_event_cb(int event)
{
    struct glimpse_data *data = NULL;

    pthread_mutex_lock(&plugin_data_lock);
    for (int i = 0; i < all_plugin_data.size(); i++) {
        if (all_plugin_data[i]->render_event_id == event) {
            data = all_plugin_data[i];
            break;
        }
    }
    // FIXME: We can't use data->log here if data is NULL...
    gm_assert(data->log, data != NULL,
              "Failed to find plugin data by event ID = %d",
              event);
    pthread_mutex_unlock(&plugin_data_lock);

#if defined(USE_GLES) || defined(USE_CORE_GL)
    /* Holding this lock while rendering implies it's not possible to start
     * terminating the plugin state during a render event callback...
     */
    pthread_mutex_lock(&life_cycle_lock);

    /* If we've already started terminating the plugin state though, then
     * we should bail immediately...
     */
    if (terminating) {
        pthread_mutex_unlock(&life_cycle_lock);
        return;
    }

    gm_debug(data->log, "Render Event %d DEBUG\n", event);

#if TARGET_OS_MAC == 0// (OSX AND IOS)
    /* We just assume Unity isn't registering a GL debug callback and
     * cross our fingers...
     */
    if (!data->registered_gl_debug_callback) {
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
        glDebugMessageCallback((GLDEBUGPROC)on_khr_debug_message_cb, data);
        data->registered_gl_debug_callback = true;
    }
#endif

    render_ar_video_background(data);

    pthread_mutex_unlock(&life_cycle_lock);
#endif // USE_GLES
}

static bool
gm_format_verify(enum gm_format glimpse_format,
                 enum UnityRenderingExtTextureFormat unity_format)
{
    switch(glimpse_format) {
    case GM_FORMAT_LUMINANCE_U8:
        switch(unity_format) {
        case kUnityRenderingExtFormatR8_SRGB:
        case kUnityRenderingExtFormatR8_UNorm:
        case kUnityRenderingExtFormatR8_SNorm:
        case kUnityRenderingExtFormatR8_UInt:
        case kUnityRenderingExtFormatR8_SInt:
            return true;
        default:
            return false;
        }

    case GM_FORMAT_RGB_U8:
        switch(unity_format) {
        case kUnityRenderingExtFormatR8G8B8_SRGB:
        case kUnityRenderingExtFormatR8G8B8_UNorm:
        case kUnityRenderingExtFormatR8G8B8_SNorm:
        case kUnityRenderingExtFormatR8G8B8_UInt:
        case kUnityRenderingExtFormatR8G8B8_SInt:
            return true;
        default:
            return false;
        }

    case GM_FORMAT_BGR_U8:
        switch(unity_format) {
        case kUnityRenderingExtFormatB8G8R8_SRGB:
        case kUnityRenderingExtFormatB8G8R8_UNorm:
        case kUnityRenderingExtFormatB8G8R8_SNorm:
        case kUnityRenderingExtFormatB8G8R8_UInt:
        case kUnityRenderingExtFormatB8G8R8_SInt:
            return true;
        default:
            return false;
        }

    case GM_FORMAT_RGBX_U8:
    case GM_FORMAT_RGBA_U8:
        switch(unity_format) {
        case kUnityRenderingExtFormatR8G8B8A8_SRGB:
        case kUnityRenderingExtFormatR8G8B8A8_UNorm:
        case kUnityRenderingExtFormatR8G8B8A8_SNorm:
        case kUnityRenderingExtFormatR8G8B8A8_UInt:
        case kUnityRenderingExtFormatR8G8B8A8_SInt:
            return true;
        default:
            return false;
        }

    case GM_FORMAT_BGRX_U8:
    case GM_FORMAT_BGRA_U8:
        switch(unity_format) {
        case kUnityRenderingExtFormatB8G8R8A8_SRGB:
        case kUnityRenderingExtFormatB8G8R8A8_UNorm:
        case kUnityRenderingExtFormatB8G8R8A8_SNorm:
        case kUnityRenderingExtFormatB8G8R8A8_UInt:
        case kUnityRenderingExtFormatB8G8R8A8_SInt:
            return true;
        default:
            return false;
        }

    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_Z_U16_MM:
    case GM_FORMAT_Z_F32_M:
    case GM_FORMAT_Z_F16_M:
    case GM_FORMAT_POINTS_XYZC_F32_M:
        return (unity_format == kUnityRenderingExtFormatNone);
    }

    return false;
}

static void texture_update_callback(int eventType, void *userdata)
{
    auto event = static_cast<UnityRenderingExtEventType>(eventType);

    if (event != kUnityRenderingExtEventUpdateTextureBegin) {
        return;
    }

    auto params = reinterpret_cast<UnityRenderingExtTextureUpdateParams*>(userdata);
    struct glimpse_data *data = NULL;
    for (int i = 0; i < all_plugin_data.size(); ++i) {
        if (all_plugin_data[i]->render_event_id == (int)params->userData) {
            data = all_plugin_data[i];
            break;
        }
    }

    if (!data || data->last_video_frame == data->texture_frame) {
        return;
    }

    pthread_mutex_lock(&data->swap_frames_lock);
    struct gm_frame *new_frame = gm_frame_ref(data->last_video_frame);
    gm_frame_add_breadcrumb(new_frame, "texture update callback");

    const struct gm_intrinsics *video_intrinsics =
        &new_frame->video_intrinsics;
    int video_width = video_intrinsics->width;
    int video_height = video_intrinsics->height;

    bool format_valid = gm_format_verify(new_frame->video_format,
                                         params->format);
    if (!format_valid ||
        (int)params->width != video_width ||
        (int)params->height != video_height) {
        if (!format_valid) {
            gm_debug(data->log,
                     "texture_update_callback: Texture format mismatch");
        } else {
            gm_debug(data->log,
                     "texture_update_callback_v2: Texture size mismatch "
                     "(%dx%d != %dx%d)",
                     (int)params->width, (int)params->height,
                     video_width, video_height);
        }
        gm_frame_add_breadcrumb(new_frame,
                                "texture update cb size mismatch discard");
        gm_frame_unref(new_frame);
        pthread_mutex_unlock(&data->swap_frames_lock);
        return;
    }

    if (data->texture_frame) {
        gm_frame_add_breadcrumb(data->texture_frame,
                                "texture update cb discard");
        gm_frame_unref(data->texture_frame);
    }
    data->texture_frame = new_frame;
    params->texData = new_frame->video->data;

    pthread_mutex_unlock(&data->swap_frames_lock);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_video_texture_update_callback(void)
{
    return texture_update_callback;
}

extern "C" const bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_video_texture_format(intptr_t plugin_handle,
                                  int *out_width, int *out_height,
                                  enum gm_format *out_format)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data || !data->last_video_frame) {
        return false;
    }

    pthread_mutex_lock(&data->swap_frames_lock);
    if (!data->last_video_frame) {
        pthread_mutex_unlock(&data->swap_frames_lock);
        return false;
    }

    const struct gm_intrinsics *video_intrinsics =
        &data->last_video_frame->video_intrinsics;
    *out_width = video_intrinsics->width;
    *out_height = video_intrinsics->height;
    *out_format = data->last_video_frame->video_format;

    pthread_mutex_unlock(&data->swap_frames_lock);

    return true;
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_render_event_callback(void)
{
    return on_render_event_cb;
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
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    gm_debug(data->log, "GLIMPSE: Context Event\n");

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
    struct glimpse_data *data = (struct glimpse_data *)user_data;

    gm_debug(data->log, "GLIMPSE: Device Event\n");

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_cond_signal(&data->event_notify_cond);
    pthread_mutex_unlock(&data->event_queue_lock);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_terminate(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }

    gm_debug(data->log, "GLIMPSE: Terminate\n");

    pthread_mutex_lock(&life_cycle_lock);
    terminating = true;
    pthread_mutex_unlock(&life_cycle_lock);

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
    if (data->device)
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

    if (data->ctx)
        gm_context_destroy(data->ctx);


    /* locking redundant here, but help serve as a reminder that
     * this state should only be touched with this lock held...
     */
    pthread_mutex_lock(&data->swap_frames_lock);

    if (data->visible_frame) {
        gm_frame_unref(data->visible_frame);
        data->visible_frame = NULL;
    }
    if (data->texture_frame) {
        gm_frame_unref(data->texture_frame);
        data->texture_frame = NULL;
    }
    if (data->last_depth_frame) {
        gm_frame_unref(data->last_depth_frame);
        data->last_depth_frame = NULL;
    }
    if (data->last_video_frame) {
        gm_frame_unref(data->last_video_frame);
        data->last_video_frame = NULL;
    }

    pthread_mutex_unlock(&data->swap_frames_lock);

    if (data->device)
        gm_device_close(data->device);

    gm_debug(data->log, "Destroying logger");
    gm_logger_destroy(data->log);
    fclose(data->log_fp);
    unity_log_function = NULL;

    json_value_free(data->config_val);

    pthread_mutex_lock(&plugin_data_lock);
    bool found = false;
    for (int i = 0; i < all_plugin_data.size(); i++) {
        if (all_plugin_data[i] == data) {
            std::swap(all_plugin_data[i], all_plugin_data.back());
            all_plugin_data.pop_back();
            found = true;
        }
    }
    gm_assert(data->log, found == true, "Failed to unregister terminated plugin data");
    pthread_mutex_unlock(&plugin_data_lock);

    delete data;
}

/* XXX: multiple calls to _init will return the same singleton plugin
 * state.
 * XXX: Only a single call to _terminate will destroy the plugin state
 * (i.e. it's not ref-counted.
 */
extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_init(char *config_json, bool force_null_device)
{
#if TARGET_OS_IOS == 1
    ios_log("gm_unity_init\n");
#endif

    struct glimpse_data *data = new glimpse_data();

    data->config_val = json_parse_string(config_json);
    data->config = json_object(data->config_val);

    data->render_event_id = next_unique_event_id++;

    pthread_mutex_lock(&plugin_data_lock);
    all_plugin_data.push_back(data);
    pthread_mutex_unlock(&plugin_data_lock);

    terminating = false;

    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

#if TARGET_OS_IOS == 1
    char *assets_root = ios_util_get_documents_path();
    char log_filename_tmp[PATH_MAX];
    snprintf(log_filename_tmp, sizeof(log_filename_tmp),
             "%s/glimpse.log", assets_root);
    data->log_fp = fopen(log_filename_tmp, "w");
#elif defined(__ANDROID__)
    // During development on Android we are manually uploading recording and
    // training models to /sdcard on test devices so that build+upload cycles
    // of packages built via Unity can be as quick as possible by reducing
    // the size of .apk we have to repeatedly upload.
    //
    char *assets_root = strdup("/sdcard/Glimpse");
    char log_filename_tmp[PATH_MAX];
    snprintf(log_filename_tmp, sizeof(log_filename_tmp),
             "%s/glimpse.log", assets_root);
    data->log_fp = fopen(log_filename_tmp, "w");
    gm_assert(data->log, android_jvm_singleton != NULL,
              "Expected to have discovered JavaVM before gm_unity_init()");
#else
    const char *assets_root_env = getenv("GLIMPSE_ASSETS_ROOT");
    char *assets_root = strdup(assets_root_env ? assets_root_env : ".");
    data->log_fp = fopen("glimpse.log", "w");

    const char *assets_path_override =
        json_object_get_string(data->config, "assetsPath");
    if (assets_path_override && strlen(assets_path_override) != 0) {
        free(assets_root);
        assets_root = strdup(assets_path_override);
    }
#endif

    gm_set_assets_root(data->log, assets_root);

    char recordings_path_tmp[PATH_MAX];
    snprintf(recordings_path_tmp, sizeof(recordings_path_tmp),
             "%s/ViewerRecording", assets_root);
    const char *recordings_path = recordings_path_tmp;

    if (!getenv("FAKENECT_PATH")) {
        char fakenect_path[PATH_MAX];
        snprintf(fakenect_path, sizeof(fakenect_path),
                 "%s/FakeRecording", assets_root);
        setenv("FAKENECT_PATH", fakenect_path, true);
    }

    gm_debug(data->log, "Init");
    gm_debug(data->log, "Config:\n%s", config_json);

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

    gm_debug(data->log, "Recording Path set to: %s", recordings_path);

    bool have_recording = false;
    char full_recording_path[512];
    const char *recording_name = json_object_get_string(data->config, "recordingName");
    if (recording_name && strlen(recording_name) > 0) {
        gm_debug(data->log, "Recording Name: %s", recording_name);

        snprintf(full_recording_path, sizeof(full_recording_path),
                 "%s/%s", recordings_path, recording_name);

        struct stat sb;
        if (stat(full_recording_path, &sb) == 0)
            have_recording = true;
    }

    pthread_mutex_init(&data->event_queue_lock, NULL);
    pthread_cond_init(&data->event_notify_cond, NULL);
    data->events_front = new std::vector<struct event>();
    data->events_back = new std::vector<struct event>();

    pthread_mutex_init(&data->swap_frames_lock, NULL);

    char *ctx_err = NULL;
    data->ctx = gm_context_new(data->log, &ctx_err);
    if (!data->ctx) {
        gm_error(data->log, "Failed to create Glimpse tracking context: %s", ctx_err);
        gm_unity_terminate((intptr_t)data);
        return NULL;
    }
    gm_context_set_event_callback(data->ctx, on_event_cb, data);

    const char *config_name = json_object_get_string(data->config, "contextConfig");
    if (config_name && strlen(config_name)) {
        char *open_err = NULL;
        struct gm_asset *config_asset =
            gm_asset_open(data->log,
                          config_name,
                          GM_ASSET_MODE_BUFFER, &open_err);
        if (config_asset) {
            const char *buf = (const char *)gm_asset_get_buffer(config_asset);
            JSON_Value *json_config = json_parse_string(buf);
            gm_context_set_config(data->ctx, json_config);
            json_value_free(json_config);
            gm_asset_close(config_asset);
        } else {
            gm_warn(data->log, "Failed to open %s: %s", config_name, open_err);
            free(open_err);
        }
    }

    struct gm_device_config config = {};

    int device_choice = json_object_get_number(data->config, "device");

#if TARGET_OS_IOS == 1
    gm_info(data->log, "Checking 'iosDevice' config");
    if (json_object_has_value(data->config, "iosDevice")) {
        device_choice = json_object_get_number(data->config, "iosDevice");
    }
#endif

    switch (device_choice) {
    case 0: // Auto
#ifdef USE_TANGO
        gm_info(data->log, "Requested device type = Auto (Tango)");
        config.type = GM_DEVICE_TANGO;
#elif TARGET_OS_IOS == 1
        gm_info(data->log, "Requested device type = Auto (Avf)");
        config.type = GM_DEVICE_AVF;
#else
        if (have_recording) {
            gm_info(data->log, "Requested device type = Auto (Recording)");
            config.type = GM_DEVICE_RECORDING;
            config.recording.path = full_recording_path;
        } else {
#ifdef USE_FREENECT
            gm_info(data->log, "Requested device type = Auto (Kinect)");
            config.type = GM_DEVICE_KINECT;
#else
            gm_info(data->log, "Requested device type = Auto (NULL)");
            config.type = GM_DEVICE_NULL;
#endif
        }
#endif
        break;
    case 1:
        gm_info(data->log, "Requested device type = Kinect");
        config.type = GM_DEVICE_KINECT;
        break;
    case 2:
        gm_info(data->log, "Requested device type = Recording");
        config.type = GM_DEVICE_RECORDING;
        config.recording.path = full_recording_path;
        break;
    case 3:
        gm_info(data->log, "Requested device type = NULL");
        config.type = GM_DEVICE_NULL;
        break;
    }

    if (force_null_device)
        config.type = GM_DEVICE_NULL;

    char *dev_err = NULL;
    data->device = gm_device_open(data->log, &config, &dev_err);
    if (!data->device) {
        gm_error(data->log, "Failed to open device: %s", dev_err);
        gm_unity_terminate((intptr_t)data);
        return NULL;
    }
    data->device_type = gm_device_get_type(data->device);
    gm_device_set_event_callback(data->device, on_device_event_cb, data);
#ifdef __ANDROID__
    gm_device_attach_jvm(data->device, android_jvm_singleton);
#endif

    if (!gm_device_commit_config(data->device, &dev_err)) {
        gm_error(data->log, "Failed to commit device configuration: %s", dev_err);
        gm_device_close(data->device);
        data->device = NULL;
        gm_unity_terminate((intptr_t)data);
        return NULL;
    }

    return (intptr_t )data;
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_render_event_id(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }

    return data->render_event_id;
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_n_joints(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }

    return gm_context_get_n_joints(data->ctx);
}

extern "C" const char * UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_joint_name(intptr_t plugin_handle, int joint_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return "Unknown";
    }

    int n_joints = gm_context_get_n_joints(data->ctx);
    if (joint_no < 0 || joint_no >= n_joints) {
        gm_error(data->log, "Out of bounds joint index %d (n_joints = %d)",
                 joint_no, n_joints);
        return "Unknown";
    }

    return gm_context_get_joint_name(data->ctx, joint_no);
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_joint_semantic(intptr_t plugin_handle, int joint_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return GM_JOINT_UNKNOWN;
    }

    int n_joints = gm_context_get_n_joints(data->ctx);
    if (joint_no < 0 || joint_no >= n_joints) {
        gm_error(data->log, "Out of bounds joint index %d (n_joints = %d)",
                 joint_no, n_joints);
        return GM_JOINT_UNKNOWN;
    }

    return (int)gm_context_get_joint_semantic(data->ctx, joint_no);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_latest_tracking(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    struct gm_tracking *tracking = gm_context_get_latest_tracking(data->ctx);

    gm_debug(data->log, "Get Latest Tracking %p", tracking);

    return (intptr_t)tracking;
}

extern "C" const bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_has_skeleton(intptr_t plugin_handle,
                               intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return false;
    }

    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;

    return gm_tracking_has_skeleton(tracking);
}

extern "C" const intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_get_skeleton(intptr_t plugin_handle,
                               intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;

    return (intptr_t)gm_tracking_get_skeleton(tracking);
}

extern "C" const uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_get_timestamp(intptr_t plugin_handle,
                                intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }

    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;

    return gm_tracking_get_timestamp(tracking);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_tracking_unref(intptr_t plugin_handle, intptr_t tracking_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }

    struct gm_tracking *tracking = (struct gm_tracking *)tracking_handle;

    int ref = tracking->ref;
    gm_debug(data->log, "Tracking Unref %p (ref = %d)",
             tracking,
             ref);
    gm_tracking_unref(tracking);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_context_get_prediction(intptr_t plugin_handle,
                                uint64_t delay)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    if (data->last_video_frame) {
        uint64_t timestamp = data->last_video_frame->timestamp - delay;

        struct gm_prediction *prediction = gm_context_get_prediction(data->ctx,
                                                                     timestamp);

        gm_debug(data->log, "Get Prediction: delay=%" PRIu64 "ns, ts=%" PRIu64 "ns: %p",
                 delay, timestamp, prediction);

        return (intptr_t)prediction;
    } else {
        return (intptr_t)NULL;
    }
}

extern "C" const intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_prediction_get_skeleton(intptr_t plugin_handle,
                                 intptr_t prediction_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }
    struct gm_prediction *prediction =
        (struct gm_prediction *)prediction_handle;

    return (intptr_t)gm_prediction_get_skeleton(prediction);
}

extern "C" const uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_prediction_get_timestamp(intptr_t plugin_handle,
                                  intptr_t prediction_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    struct gm_prediction *prediction =
        (struct gm_prediction *)prediction_handle;

    return gm_prediction_get_timestamp(prediction);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_prediction_unref(intptr_t plugin_handle, intptr_t prediction_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }
    struct gm_prediction *prediction = (struct gm_prediction *)prediction_handle;

    int ref = prediction->ref;
    gm_debug(data->log, "Prediction Unref %p (ref = %d)",
             prediction,
             ref);
    gm_prediction_unref(prediction);
}

/* XXX: deprecated, use gm_unity_context_get_n_joints() */
extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_get_n_joints(intptr_t plugin_handle,
                               intptr_t skeleton_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return 0;
    }

    return gm_skeleton_get_n_joints(skeleton);
}

extern "C" const float * UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_get_joint_position(intptr_t plugin_handle,
                                     intptr_t skeleton_handle,
                                     int joint_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return NULL;
    }
    int n_joints = gm_skeleton_get_n_joints(skeleton);
    if (joint_no < 0 || joint_no >= n_joints) {
        gm_error(data->log, "Out of bounds joint index %d (n_joints = %d)",
                 joint_no, n_joints);
        return NULL;
    }

    const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, joint_no);
    if (joint)
        return (const float *)&(joint->x);
    else
        return NULL;
}

extern "C" const bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_is_joint_valid(intptr_t plugin_handle,
                                 intptr_t skeleton_handle,
                                 int joint_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return false;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return false;
    }
    int n_joints = gm_skeleton_get_n_joints(skeleton);
    if (joint_no < 0 || joint_no >= n_joints) {
        gm_error(data->log, "Out of bounds joint index %d (n_joints = %d)",
                 joint_no, n_joints);
        return false;
    }

    return !!gm_skeleton_get_joint(skeleton, joint_no);
}

/* XXX: deprecated, use gm_unity_context_get_joint_name() */
extern "C" const char * UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_get_joint_name(intptr_t plugin_handle,
                                 intptr_t skeleton_handle,
                                 int joint_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return "";
    }

    return gm_context_get_joint_name(data->ctx, joint_no);
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_get_n_bones(intptr_t plugin_handle, intptr_t skeleton_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return 0;
    }

    return gm_skeleton_get_n_bones(skeleton);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_get_bone(intptr_t plugin_handle,
                           intptr_t skeleton_handle,
                           int bone)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return NULL;
    }

    return (intptr_t)gm_skeleton_get_bone(skeleton, bone);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_resize(intptr_t plugin_handle,
                         intptr_t skeleton_handle,
                         intptr_t ref_skeleton_handle,
                         int parent_joint)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }
    const struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return NULL;
    }
    const struct gm_skeleton *ref_skeleton = (struct gm_skeleton *)ref_skeleton_handle;
    if (!ref_skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return NULL;
    }

    return (intptr_t)gm_skeleton_resize(data->ctx, skeleton, ref_skeleton, parent_joint);
}

/* It's assumed that GlimpseRuntime knows when it has ownership of a skeleton
 * and is responsible for freeing it. Notably skeletons got via tracking and
 * prediction objects shouldn't be freed, but e.g. a skeleton derived by
 * resizing a target pose to match proportions of the person being tracked
 * may need to be freed
 */
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_skeleton_free(intptr_t plugin_handle, intptr_t skeleton_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }
    struct gm_skeleton *skeleton = (struct gm_skeleton *)skeleton_handle;
    if (!skeleton) {
        gm_error(data->log, "NULL skeleton handle");
        return;
    }

    gm_skeleton_free(skeleton);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_bone_get_head(intptr_t plugin_handle, intptr_t bone_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    const struct gm_bone *bone = (struct gm_bone *)bone_handle;
    if (!bone) {
        gm_error(data->log, "NULL bone handle");
        return NULL;
    }

    return (intptr_t)gm_bone_get_head(bone);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_bone_get_tail(intptr_t plugin_handle, intptr_t bone_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    const struct gm_bone *bone = (struct gm_bone *)bone_handle;
    if (!bone) {
        gm_error(data->log, "NULL bone handle");
        return NULL;
    }

    return (intptr_t)gm_bone_get_tail(bone);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_open(intptr_t plugin_handle,
                              const char *sequence_name)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }

    if (sequence_name == NULL) {
        gm_error(data->log, "NULL gm_unity_target_sequence_open() sequence_name");
        return NULL;
    }

    char path_tmp[PATH_MAX];
    xsnprintf(path_tmp, sizeof(path_tmp),
              "Targets/%s/glimpse_target.index",
              sequence_name);

    gm_debug(data->log, "gm_unity_target_sequence_open(), opening %s",
             sequence_name);

    char *catch_err = NULL;
    struct gm_target *target =
        gm_target_new_from_index(data->ctx,
                                 data->log,
                                 path_tmp,
                                 &catch_err);
    if (!target) {
        gm_error(data->log, "Failed to open target sequence %s: %s",
                 path_tmp, catch_err);
        free(catch_err);
        catch_err = NULL;
        return NULL;
    }

    return (intptr_t)target;
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_get_n_frames(intptr_t plugin_handle,
                                      intptr_t target_sequence)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    struct gm_target *sequence = (struct gm_target *)target_sequence;
    if (!sequence) {
        gm_error(data->log, "NULL sequence handle");
        return 0;
    }

    return gm_target_get_n_frames(sequence);
}

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_get_frame(intptr_t plugin_handle,
                                   intptr_t target_sequence)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    struct gm_target *sequence = (struct gm_target *)target_sequence;
    if (!sequence) {
        gm_error(data->log, "NULL sequence handle");
        return 0;
    }

    return gm_target_get_frame(sequence);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_set_frame(intptr_t plugin_handle,
                                   intptr_t target_sequence,
                                   int frame_no)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }
    struct gm_target *sequence = (struct gm_target *)target_sequence;
    if (!sequence) {
        gm_error(data->log, "NULL sequence handle");
        return;
    }

    gm_target_set_frame(sequence, frame_no);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_get_skeleton(intptr_t plugin_handle,
                                      intptr_t target_sequence)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return NULL;
    }
    struct gm_target *sequence = (struct gm_target *)target_sequence;
    if (!sequence) {
        gm_error(data->log, "NULL sequence handle");
        return NULL;
    }

    return (intptr_t)gm_target_get_skeleton(sequence);
}

extern "C" float UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_target_sequence_get_bone_error(intptr_t plugin_handle,
                                        intptr_t target_sequence,
                                        intptr_t bone_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return 0;
    }
    struct gm_target *sequence = (struct gm_target *)target_sequence;
    if (!sequence) {
        gm_error(data->log, "NULL sequence handle");
        return 0;
    }

    const struct gm_bone *bone = (const struct gm_bone *)bone_handle;
    if (!bone) {
        gm_error(data->log, "NULL bone handle");
        return 0;
    }

    return (intptr_t)gm_target_get_error(sequence, bone);
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

extern "C" const bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_video_projection(intptr_t plugin_handle, float *out_mat4)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return false;
    }

    if (data->last_video_frame) {
        struct gm_intrinsics *intrinsics = &data->last_video_frame->video_intrinsics;
        enum gm_rotation rotation = data->last_video_frame->camera_rotation;
        struct gm_intrinsics rotated_intrinsics;

        gm_context_rotate_intrinsics(data->ctx,
                                     intrinsics,
                                     &rotated_intrinsics,
                                     rotation);

        memcpy(out_mat4,
               glm::value_ptr(intrinsics_to_project_matrix(&rotated_intrinsics, 0.1, 100)),
               sizeof(float) * 16);
        return true;
    } else {
        return false;
    }
}

extern "C" bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_run(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return false;
    }

    if (!data->device_ready)
        return false;

    gm_debug(data->log, "GLIMPSE: Run\n");
    gm_device_start(data->device);
    gm_context_enable(data->ctx);

    return true;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_stop(intptr_t plugin_handle)
{
    struct glimpse_data *data = (struct glimpse_data *)plugin_handle;
    if (!data) {
        return;
    }

    gm_debug(data->log, "GLIMPSE: Stop\n");
    gm_context_disable(data->ctx);
    //gm_device_stop(data->device);
}

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces *interfaces)
{
#if TARGET_OS_IOS == 1
    ios_log("UnityPluginLoad");
#endif
    unity_interfaces = interfaces;
    unity_graphics = interfaces->Get<IUnityGraphics>();
    unity_graphics->RegisterDeviceEventCallback(on_graphics_device_event_cb);

    on_graphics_device_event_cb(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload(void)
{
#if TARGET_OS_IOS == 1
    ios_log("UnityPluginUnload");
#endif
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
JNI_OnLoad(JavaVM *vm, void *reserved)
{
    android_jvm_singleton = vm;

    //instantiate_glimpse_test_class(jni_env);
    return JNI_VERSION_1_6;
}
#endif
