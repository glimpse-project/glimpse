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

#undef GM_LOG_CONTEXT

#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Plugin"
#define GLSL_SHADER_VERSION "#version 300 es\n"
#include <android/log.h>
#include <jni.h>
#else
#define GM_LOG_CONTEXT "unity_plugin"
#define GLSL_SHADER_VERSION "#version 150\n"
#endif

#ifdef USE_TANGO
#include <tango_client_api.h>
#include <tango_support_api.h>
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
    JSON_Value *config_val;
    JSON_Object *config;

    FILE *log_fp;
    struct gm_logger *log;

    struct gm_context *ctx;
    struct gm_device *device;

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
    GLuint gl_vid_tex;

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

    GLuint yuv_frame_video_program;
    GLuint video_program;

    GLuint attrib_quad_bo;

    GLuint attrib_quad_pos;
    GLuint attrib_quad_tex_coords;

    GLuint uniform_tex_sampler;
};

#ifdef __ANDROID__
static JavaVM *android_jvm_singleton;
#endif

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

/* Considering that Unity renders in a separate thread which is awkward to
 * synchronize with from C# we need to take extra care while terminating
 * our plugin state because it's possible we might still see render
 * event callbacks which shouldn't try and access the plugin state
 */
static pthread_mutex_t life_cycle_lock;
static bool terminating; // checked in on_render_event_cb()
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
                                         data->last_depth_frame->timestamp,
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

        void *video_front = gm_frame_get_video_buffer(data->last_video_frame);
        enum gm_format video_format =
            gm_frame_get_video_format(data->last_video_frame);

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
handle_device_ready(struct glimpse_data *data)
{
    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(data->device);
    gm_context_set_depth_camera_intrinsics(data->ctx, depth_intrinsics);

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(data->device);
    gm_context_set_video_camera_intrinsics(data->ctx, video_intrinsics);
    /*gm_context_set_depth_to_video_camera_extrinsics(data->ctx,
      gm_device_get_depth_to_video_extrinsics(data->device));*/

    const char *config_name = json_object_get_string(data->config, "deviceConfig");
    if (config_name && strlen(config_name)) {
        char *open_err = NULL;
        struct gm_asset *config_asset =
            gm_asset_open(data->log,
                          config_name,
                          GM_ASSET_MODE_BUFFER, &open_err);
        if (config_asset) {
            const char *buf = (const char *)gm_asset_get_buffer(config_asset);
            gm_config_load(data->log, buf, gm_device_get_ui_properties(data->device));
            gm_asset_close(config_asset);
        } else {
            gm_warn(data->log, "Failed to open %s: %s", config_name, open_err);
            free(open_err);
        }
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

        /* It's always possible that we will see an event for a frame
         * that was ready before we upgraded the buffers_mask for what
         * we need, so we skip notifications for frames we can't use.
         */
        if (event->frame_ready.buffers_mask & data->pending_frame_buffers_mask)
        {
            data->pending_frame_buffers_mask = -1;
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
#ifdef USE_TANGO
            "#extension GL_OES_EGL_image_external_essl3 : require\n"
#endif
            "precision highp float;\n"
            "precision highp int;\n"
#ifdef USE_TANGO
            "uniform samplerExternalOES tex_sampler;\n"
#else
            "uniform sampler2D tex_sampler;\n"
#endif
            "in vec2 tex_coords;\n"
            "out lowp vec4 frag_color;\n"
            "void main() {\n"
            "  frag_color = texture(tex_sampler, tex_coords);\n"
            "}\n";

        data->video_program = gm_gl_create_program(data->log,
                                                   vert_shader,
                                                   frag_shader,
                                                   NULL);

        data->attrib_quad_pos =
            glGetAttribLocation(data->video_program, "pos");
        data->attrib_quad_tex_coords =
            glGetAttribLocation(data->video_program, "tex_coords_in");

        data->uniform_tex_sampler = glGetUniformLocation(data->video_program, "tex_sampler");
        glUseProgram(data->video_program);
        glUniform1i(data->uniform_tex_sampler, 0);
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

    GLenum target = GL_TEXTURE_2D;
#ifdef __ANDROID__
#ifdef USE_TANGO
    target = GL_TEXTURE_EXTERNAL_OES;
    glBindTexture(target, data->gl_vid_tex);
#else
    glBindTexture(target, data->gl_vid_tex);
#endif
#else
    glBindTextureUnit(0, data->gl_vid_tex);
#endif
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

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
    gm_assert(data->log, !!data->ctx, "render_ar_video_background, NULL ctx");

#ifndef USE_TANGO
    /* Upload latest video frame if it's changed...
     */
    if (data->last_video_frame != data->visible_frame) {
        const struct gm_intrinsics *video_intrinsics =
            gm_device_get_video_intrinsics(data->device);
        int video_width = video_intrinsics->width;
        int video_height = video_intrinsics->height;

        pthread_mutex_lock(&data->swap_frames_lock);

        struct gm_frame *new_frame = gm_frame_ref(data->last_video_frame);

        gm_frame_add_breadcrumb(new_frame, "render thread visible");

        if (data->visible_frame) {
            gm_frame_add_breadcrumb(data->visible_frame, "render thread discard");
            gm_frame_unref(data->visible_frame);
        }
        data->visible_frame = new_frame;

        pthread_mutex_unlock(&data->swap_frames_lock);

        /*
         * Update video from camera
         */
        glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        void *video_front = new_frame->video->data;
        enum gm_format video_format = new_frame->video_format;

        if (data->gl_vid_tex == 0) {
            glGenTextures(1, &data->gl_vid_tex);
            glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        switch (video_format) {
        case GM_FORMAT_LUMINANCE_U8:
            gm_debug(data->log, "uploading U8 %dx%d", video_width, video_height);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                         video_width, video_height,
                         0, GL_RED, GL_UNSIGNED_BYTE, video_front);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
            break;

        case GM_FORMAT_RGB_U8:
            gm_debug(data->log, "uploading RGB8 %dx%d", video_width, video_height);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                         video_width, video_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_RGBX_U8:
        case GM_FORMAT_RGBA_U8:
            gm_debug(data->log, "uploading RGBA8 %dx%d", video_width, video_height);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                         video_width, video_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, video_front);
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
    }
#else
    if (data->gl_vid_tex == 0) {
        glGenTextures(1, &data->gl_vid_tex);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, data->gl_vid_tex);

        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
#endif

    if (data->gl_vid_tex != 0 && data->last_video_frame != NULL) {

#ifdef USE_TANGO
        if (TangoService_updateTextureExternalOes(
                TANGO_CAMERA_COLOR, data->gl_vid_tex,
                NULL /* ignore timestamp */) != TANGO_SUCCESS) {
            gm_warn(data->log, "Failed to get a color image.");
        }
#endif

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

static void UNITY_INTERFACE_API
on_render_event_cb(int event)
{
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

    switch (event) {
    case 0:
        gm_context_render_thread_hook(plugin_data->ctx);
        break;
    case 1:
        render_ar_video_background(plugin_data);
        break;
    }

    pthread_mutex_unlock(&life_cycle_lock);
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
    pthread_mutex_unlock(&data->event_queue_lock);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_terminate(void)
{
    struct glimpse_data *data = plugin_data;

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

    json_value_free(plugin_data->config_val);

    delete plugin_data;
    plugin_data = NULL;
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_init(char *config_json)
{
    struct glimpse_data *data = new glimpse_data();

    data->config_val = json_parse_string(config_json);
    data->config = json_object(data->config_val);

    plugin_data = data;
    terminating = false;

    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

#ifdef __ANDROID__
    // During development on Android we are manually uploading recording and
    // training models to /sdcard on test devices so that build+upload cycles
    // of packages built via Unity can be as quick as possible by reducing
    // the size of .apk we have to repeatedly upload.
    //
#define ANDROID_ASSETS_ROOT "/sdcard/Glimpse"
    setenv("GLIMPSE_ASSETS_ROOT", ANDROID_ASSETS_ROOT, true);
    setenv("FAKENECT_PATH", ANDROID_ASSETS_ROOT "/FakeRecording", true);
    data->log_fp = fopen(ANDROID_ASSETS_ROOT "/glimpse.log", "w");
#else
    data->log_fp = fopen("glimpse.log", "w");
#endif

    gm_debug(data->log, "Init");
    gm_debug(data->log, "Config:\n%s", config_json);

#ifdef __ANDROID__
    gm_assert(data->log, android_jvm_singleton != NULL,
              "Expected to have discovered JavaVM before gm_unity_init()");
#endif

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

    const char *assets_path = json_object_get_string(data->config, "assetsPath");
    if (assets_path && strlen(assets_path) != 0) {
        setenv("GLIMPSE_ASSETS_ROOT", assets_path, true);
    }

    data->events_front = new std::vector<struct event>();
    data->events_back = new std::vector<struct event>();

    pthread_mutex_init(&data->swap_frames_lock, NULL);

    char *ctx_err = NULL;
    data->ctx = gm_context_new(data->log, &ctx_err);
    if (!data->ctx) {
        gm_error(data->log, "Failed to create Glimpse tracking context: %s", ctx_err);
        gm_unity_terminate();
        return 0;
    }
    gm_context_set_event_callback(data->ctx, on_event_cb, plugin_data);

    const char *config_name = json_object_get_string(data->config, "contextConfig");
    if (config_name && strlen(config_name)) {
        char *open_err = NULL;
        struct gm_asset *config_asset =
            gm_asset_open(data->log,
                          config_name,
                          GM_ASSET_MODE_BUFFER, &open_err);
        if (config_asset) {
            const char *buf = (const char *)gm_asset_get_buffer(config_asset);
            gm_config_load(data->log, buf, gm_context_get_ui_properties(data->ctx));
            gm_asset_close(config_asset);
        } else {
            gm_warn(data->log, "Failed to open %s: %s", config_name, open_err);
            free(open_err);
        }
    }

    struct gm_device_config config = {};

    const char *recordings_path = json_object_get_string(data->config, "recordingsPath");
    if (recordings_path == NULL || strlen(recordings_path) == 0)
        recordings_path = getenv("GLIMPSE_RECORDING_PATH");
    if (!recordings_path)
        recordings_path = getenv("GLIMPSE_ASSETS_ROOT");
    if (!recordings_path)
        recordings_path = "glimpse_viewer_recording";

    const char *recording_name = json_object_get_string(data->config, "recordingName");

    char full_recording_path[512];
    snprintf(full_recording_path, sizeof(full_recording_path),
             "%s/%s", recordings_path, recording_name);

    struct stat sb;
    bool have_recording = false;
    if (stat(full_recording_path, &sb) == 0)
        have_recording = true;

    int device_choice = json_object_get_number(data->config, "device");
    switch (device_choice) {
    case 0: // Auto
#ifdef USE_TANGO
        config.type = GM_DEVICE_TANGO;
#else
        if (have_recording) {
            config.type = GM_DEVICE_RECORDING;
            config.recording.path = full_recording_path;
        } else {
            config.type = GM_DEVICE_KINECT;
        }
#endif
        break;
    case 1:
        config.type = GM_DEVICE_KINECT;
        break;
    case 2:
        config.type = GM_DEVICE_RECORDING;
        config.recording.path = full_recording_path;
        break;
    }

    char *dev_err = NULL;
    data->device = gm_device_open(data->log, &config, &dev_err);
    if (!data->device) {
        gm_error(data->log, "Failed to open device: %s", dev_err);
        gm_unity_terminate();
        return 0;
    }
    gm_device_set_event_callback(data->device, on_device_event_cb, plugin_data);
#ifdef __ANDROID__
    gm_device_attach_jvm(data->device, android_jvm_singleton);
#endif

    if (!gm_device_commit_config(data->device, &dev_err)) {
        gm_error(data->log, "Failed to commit device configuration: %s", dev_err);
        gm_device_close(data->device);
        data->device = NULL;
        gm_unity_terminate();
        return 0;
    }

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
           glm::value_ptr(intrinsics_to_project_matrix(intrinsics, 0.1, 10)),
           sizeof(float) * 16);
}

extern "C" bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_run(void)
{
    struct glimpse_data *data = plugin_data;

    if (!data->device_ready)
        return false;

    gm_debug(data->log, "GLIMPSE: Run\n");
    gm_device_start(data->device);
    gm_context_enable(data->ctx);

    return true;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_stop(void)
{
    struct glimpse_data *data = plugin_data;

    gm_debug(data->log, "GLIMPSE: Stop\n");
    gm_context_disable(data->ctx);
    //gm_device_stop(data->device);
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
JNI_OnLoad(JavaVM *vm, void *reserved)
{
    android_jvm_singleton = vm;

    //instantiate_glimpse_test_class(jni_env);
    return JNI_VERSION_1_6;
}
#endif
