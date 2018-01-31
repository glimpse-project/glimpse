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
draw_textured_quad(struct glimpse_data *data,
                   float s0, float t0,
                   float s1, float t1)
{
    struct {
        float x, y, s, t;
    } quad_strip[4] = {
#if 0
        { -1,  1, s0, t0, }, //  0  2
        { -1, -1, s0, t1, }, //  | /|
        {  1,  1, s1, t0, }, //  |/ |
        {  1, -1, s1, t1  }  //  1  3
#else
        { -1,  1, s0, t0, }, //  0  2
        { -1,  0, s0, t1, }, //  | /|
        {  1,  1, s1, t0, }, //  |/ |
        {  1,  0, s1, t1  }  //  1  3
#endif
    };

    if (!data->attrib_quad_rot_scale_bo) {
        glGenBuffers(1, &data->attrib_quad_rot_scale_bo);
        glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);

        gm_debug(data->log, "Created quad attribute buffer");

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if (!data->scale_program) {
#if 0
        const char *vert_shader =
            "precision mediump float;\n"
            "precision mediump int;\n"
            "attribute vec2 pos;\n"
            "attribute vec2 tex_coords_in;\n"
            "varying vec2 tex_coords;\n"
            "void main() {\n"
            "  tex_coords = tex_coords_in;\n"
            "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
            "}\n";
        const char *frag_shader =
            "precision highp float;\n"
            "precision highp int;\n"
            "uniform sampler2D texture;\n"
            "varying vec2 tex_coords;\n"
            "void main() {\n"
            "  gl_FragColor = texture2D(texture, tex_coords);\n"
            "}\n";

        data->scale_program = gm_gl_create_program(data->log, vert_shader,
                                                   frag_shader, NULL);

        data->attrib_quad_rot_scale_pos =
            glGetAttribLocation(data->scale_program, "pos");
        data->attrib_quad_rot_scale_tex_coords =
            glGetAttribLocation(data->scale_program, "tex_coords_in");
        data->uniform_tex_sampler = glGetUniformLocation(data->scale_program, "texture");

        glUseProgram(data->scale_program);

        glUniform1i(data->uniform_tex_sampler, 0);

        gm_debug(data->log, "Created scale shader");
#else
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
            "  frag_color = vec4(0.0, 1.0, 0.0, 1.0);\n"
            //"  frag_color = texture(tex_sampler, tex_coords);\n"
            "}\n";

        data->scale_program = gm_gl_create_program(data->log,
                                                   vert_shader,
                                                   frag_shader,
                                                   NULL);

        data->attrib_quad_rot_scale_pos =
            glGetAttribLocation(data->scale_program, "pos");
        data->attrib_quad_rot_scale_tex_coords =
            glGetAttribLocation(data->scale_program, "tex_coords_in");

        data->uniform_tex_sampler = glGetUniformLocation(data->scale_program, "tex_sampler");
        glUseProgram(data->scale_program);
        glUniform1i(data->uniform_tex_sampler, 0);

        gm_debug(data->log, "Created scale shader");
#endif
    }

    glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);
    glEnableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glVertexAttribPointer(data->attrib_quad_rot_scale_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
    glEnableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);
    glVertexAttribPointer(data->attrib_quad_rot_scale_tex_coords,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)8);


    //glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    //glBindTexture(GL_TEXTURE_2D, 0);

    glDisableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glDisableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    return;
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glEnableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glVertexAttribPointer(data->attrib_quad_rot_scale_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
    glEnableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);
    glVertexAttribPointer(data->attrib_quad_rot_scale_tex_coords,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)8);

    glUseProgram(data->scale_program);
    glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glDisableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

#if 0
    glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);

    glEnableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glVertexAttribPointer(data->attrib_quad_rot_scale_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
    glEnableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);
    glVertexAttribPointer(data->attrib_quad_rot_scale_tex_coords,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)8);


    glUseProgram(data->scale_program);
    glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    glDisableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glDisableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}

#if 0
void
render_foo(struct glimpse_data *data)
{
    struct {
        float x, y, s, t;
    } quad_strip[] = {
        { -1,  1, 0, 1, }, //  0  2
        { -1, -1, 0, 0, }, //  | /|
        {  1,  1, 1, 1, }, //  |/ |
        {  1, -1, 1, 0  }  //  1  3
    };

    uint64_t start, end, duration_ns;

    if (!data->attrib_quad_rot_scale_bo) {
        glGenBuffers(1, &data->attrib_quad_rot_scale_bo);
        glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);

        glGenBuffers(1, &data->attrib_quad_rot_scale_bo);
        glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);
    }

    /* Our first draw call will combine downsampling and rotation... */
    glBindBuffer(GL_ARRAY_BUFFER, data->attrib_quad_rot_scale_bo);

    bool need_portrait_downsample_fb;
    enum gm_rotation display_rotation = GM_ROTATION_0;

    if (display_rotation != data->current_attrib_bo_rotation) {
        float coords[] = { 0, 1, 0, 0, 1, 1, 1, 0 };
        float out_coords[8];

        //FIXME
        //TangoSupport_getVideoOverlayUVBasedOnDisplayRotation(
        //    coords, display_rotation_, out_coords);

        quad_strip[0].s = out_coords[0];
        quad_strip[0].t = out_coords[1];
        quad_strip[1].s = out_coords[2];
        quad_strip[1].t = out_coords[3];
        quad_strip[2].s = out_coords[4];
        quad_strip[2].t = out_coords[5];
        quad_strip[3].s = out_coords[6];
        quad_strip[3].t = out_coords[7];

        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);

        switch(display_rotation) {
        case GM_ROTATION_0:
            need_portrait_downsample_fb = true;
            gm_debug(data->log, "> rotation = 0");
            break;
        case GM_ROTATION_90:
            need_portrait_downsample_fb = false;
            gm_debug(data->log, "> rotation = 90");
            break;
        case GM_ROTATION_180:
            need_portrait_downsample_fb = true;
            gm_debug(data->log, "> rotation = 180");
            break;
        case GM_ROTATION_270:
            need_portrait_downsample_fb = false;
            gm_debug(data->log, "> rotation = 270");
            break;
        }
        data->current_attrib_bo_rotation = display_rotation;
    } else {
        need_portrait_downsample_fb = data->have_portrait_downsample_fb;
    }

    long rotated_frame_width, rotated_frame_height;

    if (need_portrait_downsample_fb) {
        rotated_frame_width = data->grey_height;
        rotated_frame_height = data->grey_width;
    } else {
        rotated_frame_width = data->grey_width;
        rotated_frame_height = data->grey_height;
    }
    gm_debug(data->log, "rotated frame width = %d, height = %d",
         (int)rotated_frame_width, (int)rotated_frame_height);

    if (need_portrait_downsample_fb != data->have_portrait_downsample_fb) {
        if (data->downsample_fbo) {
            gm_debug(data->log, "Discarding previous downsample fbo and texture");
            glDeleteFramebuffers(1, &data->downsample_fbo);
            data->downsample_fbo = 0;
            glDeleteTextures(1, &data->downsample_tex2d);
            data->downsample_tex2d = 0;
        }
        if (data->read_back_fbo) {
            gm_debug(data->log, "Discarding previous read_back_fbo and texture");
            glDeleteFramebuffers(1, &data->read_back_fbo);
            data->read_back_fbo = 0;
        }
    }

    if (!data->downsample_fbo) {
        gm_debug(data->log, "Allocating new %dx%d downsample fbo + texture",
             (int)(rotated_frame_width / 2),
             (int)(rotated_frame_height / 2));

        glGenFramebuffers(1, &data->downsample_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, data->downsample_fbo);

        glGenTextures(1, &downsample_tex2d_);
        glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexStorage2D(GL_TEXTURE_2D,
                       2, /* num levels */
                       GL_R8,
                       rotated_frame_width / 2, rotated_frame_height / 2);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, downsample_tex2d_, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            LOGE("Framebuffer complete check (for downsample fbo) failed");

        have_portrait_downsample_fb_ = need_portrait_downsample_fb;
    }


    /* This extension lets us avoid redundant YUV->RGB color space conversion
     * since we only care about the luminance...
     */
#ifdef USE_GL_EXT_YUV_TARGET_EXT

    if (!yuv_frame_scale_program_) {
        const char *vert_shader =
            "#version 300 es\n"
            "precision mediump float;\n"
            "precision mediump int;\n"
            "in vec2 pos;\n"
            "in vec2 tex_coords_in;\n"
            "out vec2 tex_coords;\n"
            "void main() {\n"
            "  tex_coords = tex_coords_in;\n"
            "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
            "}\n";
        /* NB: from EXT_YUV_target the red component will correspond to
         * the Y component when sampling here...
         */
        const char *frag_shader =
            "#version 300 es\n"
            "#extension GL_EXT_YUV_target : enable\n"
            "precision highp float;\n"
            "precision highp int;\n"
            "uniform __samplerExternal2DY2YEXT yuv_tex_sampler;\n"
            "in vec2 tex_coords;\n"
            "out vec4 frag_color;\n"
            "void main() {\n"
            "  frag_color = vec4(texture(yuv_tex_sampler, tex_coords).r, 0.0, 0.0, 1.0);\n"
            "}\n";

        yuv_frame_scale_program_ = gm_gl_create_program(ctx,
                                                        vert_shader,
                                                        frag_shader,
                                                        NULL);

        data->attrib_quad_rot_scale_pos = glGetAttribLocation(yuv_frame_scale_program_, "pos");
        data->attrib_quad_rot_scale_tex_coords = glGetAttribLocation(yuv_frame_scale_program_, "tex_coords_in");
        uniform_tex_sampler_ = glGetUniformLocation(yuv_frame_scale_program_, "yuv_tex_sampler");

        glUseProgram(yuv_frame_scale_program_);

        glUniform1i(uniform_tex_sampler_, 0);

        gm_debug(data->log, "Created level0 scale shader");
    }

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, video_overlay_->GetTextureId());

    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glUseProgram(yuv_frame_scale_program_);
#else
    if (!scale_program_) {
        const char *vert_shader =
            "precision mediump float;\n"
            "precision mediump int;\n"
            "attribute vec2 pos;\n"
            "attribute vec2 tex_coords_in;\n"
            "varying vec2 tex_coords;\n"
            "void main() {\n"
            "  tex_coords = tex_coords_in;\n"
            "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
            "}\n";
        const char *frag_shader =
            "precision highp float;\n"
            "precision highp int;\n"
            "uniform sampler2D texture;\n"
            "varying vec2 tex_coords;\n"
            "void main() {\n"
            "  gl_FragColor = texture2D(texture, tex_coords);\n"
            "}\n";

        data->scale_program = gm_gl_create_program(ctx->log,
                                                   vert_shader,
                                                   frag_shader,
                                                   NULL);

        data->attrib_quad_rot_scale_pos =
            glGetAttribLocation(scale_program_, "pos");
        data->attrib_quad_rot_scale_tex_coords =
            glGetAttribLocation(scale_program_, "tex_coords_in");
        uniform_tex_sampler_ = glGetUniformLocation(scale_program_, "texture");

        glUseProgram(scale_program_);

        glUniform1i(uniform_tex_sampler_, 0);

        gm_debug(data->log, "Created scale shader");
    }

    if (!data->cam_tex) {
        glGenTextures(1, &data->cam_tex);
        glBindTexture(GL_TEXTURE_2D, data->cam_tex);
        glTexStorage2D(GL_TEXTURE_2D,
                       1, /* num levels */
                       GL_R8,
                       data->grey_width, data->grey_height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    start = get_time();
    glBindTexture(GL_TEXTURE_2D, data->cam_tex);

    for (int y = 0; y < 40; y++) {
        uint8_t *p = ((uint8_t *)data->grey_buffer_1_1.data()) + data->grey_width * y;
        memset(p, 0x80, data->grey_width / 2);
    }
    for (int y = 80; y < (data->grey_height / 2); y++) {
        uint8_t *p = ((uint8_t *)data->grey_buffer_1_1.data()) + data->grey_width * y;
        memset(p, 0x80, 40);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D,
                    0, /* level */
                    0, 0, /* x_off/y_off */
                    data->grey_width, data->grey_height,
                    GL_RED,
                    GL_UNSIGNED_BYTE, data->grey_buffer_1_1.data());

    end = get_time();
    duration_ns = end - start;

    gm_debug(data->log, "Uploaded top level luminance texture to GPU via glTexSubImage2D in %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    glUseProgram(scale_program_);
#endif

    start = get_time();

    glEnableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glVertexAttribPointer(data->attrib_quad_rot_scale_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
    glEnableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);
    glVertexAttribPointer(data->attrib_quad_rot_scale_tex_coords,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)8);


    /* XXX: Note at this point the appropriate source texture (either an
     * OES external image or TEXTURE_2D) has been bound as well as a
     * suitable shader for sampling the texture and vertex attributes.
     *
     * Now we combine downsampling with the appropriate rotation to match
     * the device orientation.
     */

    /* we should target a texture for level0 and the use glGenerateMipmap
    */
    glBindFramebuffer(GL_FRAMEBUFFER, data->downsample_fbo);

    //gm_debug(data->log, "Allocated pyramid level texture + fbo in %.3f%s",
    //     get_duration_ns_print_scale(duration_ns),
    //     get_duration_ns_print_scale_suffix(duration_ns));

    glViewport(0, 0, rotated_frame_width / 2, rotated_frame_height / 2);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisableVertexAttribArray(data->attrib_quad_rot_scale_pos);
    glDisableVertexAttribArray(data->attrib_quad_rot_scale_tex_coords);

    glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

    end = get_time();
    duration_ns = end - start;

    gm_debug(data->log, "Submitted level0 downsample in %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

#if 0
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#endif

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);

    start = get_time();
    glGenerateMipmap(GL_TEXTURE_2D);

    glFinish();
    end = get_time();
    duration_ns = end - start;

    gm_debug(data->log, "glGenerateMipmap took %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    glBindTexture(GL_TEXTURE_2D, 0);

    if (!data->read_back_fbo) {
        glGenFramebuffers(1, &data->read_back_fbo);
        glGenBuffers(1, &data->read_back_pbo);

        glBindFramebuffer(GL_FRAMEBUFFER, data->read_back_fbo);
        glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               downsample_tex2d_, 1);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            LOGE("Famebuffer complete check failed");

        //gm_debug(data->log, "Allocated pyramid level texture + fbo in %.3f%s",
        //     get_duration_ns_print_scale(duration_ns),
        //     get_duration_ns_print_scale_suffix(duration_ns));

        glBindBuffer(GL_PIXEL_PACK_BUFFER, data->read_back_pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER,
                     (rotated_frame_width / 4) * (rotated_frame_height / 4),
                     nullptr, GL_DYNAMIC_READ);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, data->read_back_fbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, data->read_back_pbo);

    /* TODO: hide latency of GPU work completing by deferring read until
     * some frames later (ideally based on an explicit fence letting
     * us know when the work is done)
     *
     * TODO: investigate what options we might have on Android for a
     * zero copy path not needing a ReadPixels into a PBO.
     */
    start = get_time();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0,
                 rotated_frame_width / 4,
                 rotated_frame_height / 4,
                 GL_RED_EXT, GL_UNSIGNED_BYTE, 0);
    end = get_time();
    duration_ns = end - start;

    gm_debug(data->log, "glReadPixels took %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    start = get_time();
    void *pbo_ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0,
                                     ((rotated_frame_width / 4) *
                                      (rotated_frame_height / 4)),
                                     GL_MAP_READ_BIT);
    end = get_time();
    duration_ns = end - start;

    gm_debug(data->log, "glMapBufferRange took %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    {
        dlib::timing::timer lv0_cpy_timer("Copied pyramid level0 frame for face detection from PBO in");

        tracking->face_detect_buf_width = rotated_frame_width / 4;
        tracking->face_detect_buf_height = rotated_frame_height / 4;

        /* TODO: avoid copying out of the PBO later (assuming we can get a
         * cached mapping)
         */
        gm_debug(data->log, "face detect scratch width = %d, height = %d",
             (int)tracking->face_detect_buf_width,
             (int)tracking->face_detect_buf_height);
        data->grey_face_detect_scratch.resize(tracking->face_detect_buf_width * tracking->face_detect_buf_height);
        memcpy(data->grey_face_detect_scratch.data(), pbo_ptr, data->grey_face_detect_scratch.size());

        tracking->face_detect_buf = data->grey_face_detect_scratch.data();
        gm_debug(data->log, "tracking->face_detect_buf = %p", tracking->face_detect_buf);
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

}

static void
render_stuff(struct glimpse_data *data)
{
    struct {
        float x, y, s, t;
    } quad_strip[] = {
        { -1,  1, 0, 1, }, //  0  2
        { -1, -1, 0, 0, }, //  | /|
        {  1,  1, 1, 1, }, //  |/ |
        {  1, -1, 1, 0  }  //  1  3
    };

#if 0
    if (!is_gl_initialized_ || !is_service_connected_) {
        return;
    }

    // We need to make sure that we update the texture associated with the color
    // image.
    if (TangoService_updateTextureExternalOes(
            TANGO_CAMERA_COLOR, video_overlay_->GetTextureId(),
            &last_gpu_timestamp_) != TANGO_SUCCESS) {
        LOGE("GlimpseDemo: Failed to get a color image.");
        return;
    }
#endif

    /*
     * FIXME: don't just open code all this GL cruft here
     */

    if (need_new_scaled_frame_) {
        uint64_t start, end, duration_ns;
        long rotated_frame_width, rotated_frame_height;
        bool need_portrait_downsample_fb;

        /* FIXME: how can we query the info (namely the downsampling rate) at
         * runtime from the detector_.scanner.pyramid_type, instead of hard
         * coding...
         */
        dlib::pyramid_down<6> pyr;

#ifndef UPLOAD_CAMERA_LEVEL0_VIA_TEXIMAGE2D
        pthread_mutex_lock(&luminance_cond_mutex_);

        gm_debug(data->log, "Waiting for new camera frame to downsample");
        need_new_luminance_cam_frame_ = true;
        while (need_new_luminance_cam_frame_)
            pthread_cond_wait(&luminance_available_cond_, &luminance_cond_mutex_);
        pthread_mutex_unlock(&luminance_cond_mutex_);
#endif


        /* The most practical way to get face detection running at any half
         * decent rate on a relatively underpowered device is to massively
         * downscale the level zero image that we run the face detection
         * against.
         *
         * Once we're down to ~270x480 then DLib's face detection with just
         * one 80x80 HOG (front-facing) seems to run in a reasonable time.
         * 
         * Currently assuming a 1920x1080 initial frame, we use one special
         * case fragment shader to downsample by 50% being careful to avoid
         * redundant YUV->RGB conversion and just keeping the luminance.
         *
         * After that we'll just assume that the driver's implementation
         * of glGenerateMipmap is optimal and downsampling by another 50%.
         */
#ifdef DOWNSAMPLE_ON_GPU

        gm_debug(data->log, "Downsampling via GLES");

        if (!attrib_quad_rot_scale_bo_) {
            glGenBuffers(1, &attrib_quad_rot_scale_bo_);
            glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);
            tango_gl::util::CheckGlError("glBufferData - quad_strip");

            glGenBuffers(1, &attrib_quad_rot_scale_bo_);
            glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);
        }

        /* Our first draw call will combine downsampling and rotation... */
        glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);

        if (display_rotation_ != current_attrib_bo_rotation_) {
            gm_debug(data->log, "Orientation change to account for with face detection");
            float coords[] = { 0, 1, 0, 0, 1, 1, 1, 0 };
            float out_coords[8];

            TangoSupport_getVideoOverlayUVBasedOnDisplayRotation(
                coords, display_rotation_, out_coords);

            quad_strip[0].s = out_coords[0];
            quad_strip[0].t = out_coords[1];
            quad_strip[1].s = out_coords[2];
            quad_strip[1].t = out_coords[3];
            quad_strip[2].s = out_coords[4];
            quad_strip[2].t = out_coords[5];
            quad_strip[3].s = out_coords[6];
            quad_strip[3].t = out_coords[7];

            glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);
            tango_gl::util::CheckGlError("glBufferData - rotated tex. quad_strip");

            switch(display_rotation_) {
            case ROTATION_0:
                need_portrait_downsample_fb = true;
                gm_debug(data->log, "> rotation = 0");
                break;
            case ROTATION_90:
                need_portrait_downsample_fb = false;
                gm_debug(data->log, "> rotation = 90");
                break;
            case ROTATION_180:
                need_portrait_downsample_fb = true;
                gm_debug(data->log, "> rotation = 180");
                break;
            case ROTATION_270:
                need_portrait_downsample_fb = false;
                gm_debug(data->log, "> rotation = 270");
                break;
            }
            current_attrib_bo_rotation_ = display_rotation_;
        } else {
            need_portrait_downsample_fb = have_portrait_downsample_fb_;
        }


        if (need_portrait_downsample_fb) {
            rotated_frame_width = grey_height_;
            rotated_frame_height = grey_width_;
        } else {
            rotated_frame_width = grey_width_;
            rotated_frame_height = grey_height_;
        }
        gm_debug(data->log, "rotated frame width = %d, height = %d",
             (int)rotated_frame_width, (int)rotated_frame_height);

        if (need_portrait_downsample_fb != have_portrait_downsample_fb_) {
            if (downsample_fbo_) {
                gm_debug(data->log, "Discarding previous downsample fbo and texture");
                glDeleteFramebuffers(1, &downsample_fbo_);
                downsample_fbo_ = 0;
                glDeleteTextures(1, &downsample_tex2d_);
                downsample_tex2d_ = 0;
            }
            if (read_back_fbo_) {
                gm_debug(data->log, "Discarding previous read_back_fbo and texture");
                glDeleteFramebuffers(1, &read_back_fbo_);
                read_back_fbo_ = 0;
            }
        }

        if (!downsample_fbo_) {
            gm_debug(data->log, "Allocating new %dx%d downsample fbo + texture",
                 (int)(rotated_frame_width / 2),
                 (int)(rotated_frame_height / 2));

            glGenFramebuffers(1, &downsample_fbo_);
            tango_gl::util::CheckGlError("glGenFramebuffers downsample_fbo_");

            glBindFramebuffer(GL_FRAMEBUFFER, downsample_fbo_);
            tango_gl::util::CheckGlError("glBindFramebuffer downsample_fbo_");

            glGenTextures(1, &downsample_tex2d_);
            tango_gl::util::CheckGlError("glGenTextures downsample_tex2d_");

            glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);
            tango_gl::util::CheckGlError("glBindTexture downsample_tex2d_");

            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            glTexStorage2D(GL_TEXTURE_2D,
                           2, /* num levels */
                           GL_R8,
                           rotated_frame_width / 2, rotated_frame_height / 2);
            tango_gl::util::CheckGlError("glTexStorage2D");

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, downsample_tex2d_, 0);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                LOGE("Framebuffer complete check (for downsample fbo) failed");

            tango_gl::util::CheckGlError("glCheckFramebufferStatus");

            have_portrait_downsample_fb_ = need_portrait_downsample_fb;
        }


        /* This extension lets us avoid redundant YUV->RGB color space conversion
         * since we only care about the luminance...
         */
#ifdef USE_GL_EXT_YUV_TARGET_EXT

        if (!yuv_frame_scale_program_) {
            const char *vert_shader =
                "#version 300 es\n"
                "precision mediump float;\n"
                "precision mediump int;\n"
                "in vec2 pos;\n"
                "in vec2 tex_coords_in;\n"
                "out vec2 tex_coords;\n"
                "void main() {\n"
                "  tex_coords = tex_coords_in;\n"
                "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
                "}\n";
            /* NB: from EXT_YUV_target the red component will correspond to
             * the Y component when sampling here...
             */
            const char *frag_shader =
                "#version 300 es\n"
                "#extension GL_EXT_YUV_target : enable\n"
                "precision highp float;\n"
                "precision highp int;\n"
                "uniform __samplerExternal2DY2YEXT yuv_tex_sampler;\n"
                "in vec2 tex_coords;\n"
                "out vec4 frag_color;\n"
                "void main() {\n"
                "  frag_color = vec4(texture(yuv_tex_sampler, tex_coords).r, 0.0, 0.0, 1.0);\n"
                "}\n";

            yuv_frame_scale_program_ = tango_gl::util::CreateProgram(vert_shader, frag_shader);

            attrib_quad_rot_scale_pos_ = glGetAttribLocation(yuv_frame_scale_program_, "pos");
            tango_gl::util::CheckGlError("locate lv0 pos attribute");
            attrib_quad_rot_scale_tex_coords_ = glGetAttribLocation(yuv_frame_scale_program_, "tex_coords_in");
            tango_gl::util::CheckGlError("locate lv0 tex coord attribute");
            uniform_tex_sampler_ = glGetUniformLocation(yuv_frame_scale_program_, "yuv_tex_sampler");
            tango_gl::util::CheckGlError("locate lv0 texture uniform");

            glUseProgram(yuv_frame_scale_program_);
            tango_gl::util::CheckGlError("use program lv0 scale");

            glUniform1i(uniform_tex_sampler_, 0);
            tango_gl::util::CheckGlError("glUniformi unform_lv0_tex_");

            gm_debug(data->log, "Created level0 scale shader");
        }

        glBindTexture(GL_TEXTURE_EXTERNAL_OES, video_overlay_->GetTextureId());

        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glUseProgram(yuv_frame_scale_program_);
#else
        if (!scale_program_) {
            const char *vert_shader =
                "precision mediump float;\n"
                "precision mediump int;\n"
                "attribute vec2 pos;\n"
                "attribute vec2 tex_coords_in;\n"
                "varying vec2 tex_coords;\n"
                "void main() {\n"
                "  tex_coords = tex_coords_in;\n"
                "  gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\n"
                "}\n";
            const char *frag_shader =
                "precision highp float;\n"
                "precision highp int;\n"
                "uniform sampler2D texture;\n"
                "varying vec2 tex_coords;\n"
                "void main() {\n"
                "  gl_FragColor = texture2D(texture, tex_coords);\n"
                "}\n";

            scale_program_ = tango_gl::util::CreateProgram(vert_shader, frag_shader);

            attrib_quad_rot_scale_pos_ =
                glGetAttribLocation(scale_program_, "pos");
            tango_gl::util::CheckGlError("locate pos attribute");
            attrib_quad_rot_scale_tex_coords_ =
                glGetAttribLocation(scale_program_, "tex_coords_in");
            tango_gl::util::CheckGlError("locate tex coord attribute");
            uniform_tex_sampler_ = glGetUniformLocation(scale_program_, "texture");
            tango_gl::util::CheckGlError("locate texture uniform");

            glUseProgram(scale_program_);
            tango_gl::util::CheckGlError("use program scale");

            glUniform1i(uniform_tex_sampler_, 0);
            tango_gl::util::CheckGlError("glUniformi unform_tex_");

            gm_debug(data->log, "Created scale shader");
        }

        if (!cam_tex_) {
            glGenTextures(1, &cam_tex_);
            glBindTexture(GL_TEXTURE_2D, cam_tex_);
            glTexStorage2D(GL_TEXTURE_2D,
                           1, /* num levels */
                           GL_R8,
                           grey_width_, grey_height_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        start = get_time();
        glBindTexture(GL_TEXTURE_2D, cam_tex_);

        for (int y = 0; y < 40; y++) {
            uint8_t *p = ((uint8_t *)grey_buffer_1_1_.data()) + grey_width_ * y;
            memset(p, 0x80, grey_width_ / 2);
        }
        for (int y = 80; y < (grey_height_ / 2); y++) {
            uint8_t *p = ((uint8_t *)grey_buffer_1_1_.data()) + grey_width_ * y;
            memset(p, 0x80, 40);
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0, /* level */
                        0, 0, /* x_off/y_off */
                        grey_width_, grey_height_,
                        GL_RED,
                        GL_UNSIGNED_BYTE, grey_buffer_1_1_.data());

        tango_gl::util::CheckGlError("glTexImage2D");
        end = get_time();
        duration_ns = end - start;

        gm_debug(data->log, "Uploaded top level luminance texture to GPU via glTexSubImage2D in %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

        glUseProgram(scale_program_);
#endif

        start = get_time();

        glEnableVertexAttribArray(attrib_quad_rot_scale_pos_);
        glVertexAttribPointer(attrib_quad_rot_scale_pos_,
                              2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
        glEnableVertexAttribArray(attrib_quad_rot_scale_tex_coords_);
        glVertexAttribPointer(attrib_quad_rot_scale_tex_coords_,
                              2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)8);


        /* XXX: Note at this point the appropriate source texture (either an
         * OES external image or TEXTURE_2D) has been bound as well as a
         * suitable shader for sampling the texture and vertex attributes.
         *
         * Now we combine downsampling with the appropriate rotation to match
         * the device orientation.
         */

        /* we should target a texture for level0 and the use glGenerateMipmap
        */
        glBindFramebuffer(GL_FRAMEBUFFER, downsample_fbo_);

        //gm_debug(data->log, "Allocated pyramid level texture + fbo in %.3f%s",
        //     get_duration_ns_print_scale(duration_ns),
        //     get_duration_ns_print_scale_suffix(duration_ns));

        glViewport(0, 0, rotated_frame_width / 2, rotated_frame_height / 2);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        tango_gl::util::CheckGlError("glClear");

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        tango_gl::util::CheckGlError("glDrawArrays");

        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisableVertexAttribArray(attrib_quad_rot_scale_pos_);
        glDisableVertexAttribArray(attrib_quad_rot_scale_tex_coords_);

        glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

        end = get_time();
        duration_ns = end - start;

        gm_debug(data->log, "Submitted level0 downsample in %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

#if 0
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#endif

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);

        start = get_time();
        glGenerateMipmap(GL_TEXTURE_2D);
        tango_gl::util::CheckGlError("glGenerateMipmap");

        glFinish();
        tango_gl::util::CheckGlError("glFinish");
        end = get_time();
        duration_ns = end - start;

        gm_debug(data->log, "glGenerateMipmap took %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

        glBindTexture(GL_TEXTURE_2D, 0);

        if (!read_back_fbo_) {
            glGenFramebuffers(1, &read_back_fbo_);
            tango_gl::util::CheckGlError("glGenFramebuffers");
            glGenBuffers(1, &read_back_pbo_);
            tango_gl::util::CheckGlError("glGenBuffers");

            glBindFramebuffer(GL_FRAMEBUFFER, read_back_fbo_);
            glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                   downsample_tex2d_, 1);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                LOGE("Famebuffer complete check failed");

            tango_gl::util::CheckGlError("glCheckFramebufferStatus");

            //gm_debug(data->log, "Allocated pyramid level texture + fbo in %.3f%s",
            //     get_duration_ns_print_scale(duration_ns),
            //     get_duration_ns_print_scale_suffix(duration_ns));

            glBindBuffer(GL_PIXEL_PACK_BUFFER, read_back_pbo_);
            glBufferData(GL_PIXEL_PACK_BUFFER,
                         (rotated_frame_width / 4) * (rotated_frame_height / 4),
                         nullptr, GL_DYNAMIC_READ);
            tango_gl::util::CheckGlError("glBufferData");

            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, read_back_fbo_);
        tango_gl::util::CheckGlError("glBindFramebuffer");
        glBindBuffer(GL_PIXEL_PACK_BUFFER, read_back_pbo_);
        tango_gl::util::CheckGlError("glBindBuffer");

        /* TODO: hide latency of GPU work completing by deferring read until
         * some frames later (ideally based on an explicit fence letting
         * us know when the work is done)
         *
         * TODO: investigate what options we might have on Android for a
         * zero copy path not needing a ReadPixels into a PBO.
         */
        start = get_time();
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0,
                     rotated_frame_width / 4,
                     rotated_frame_height / 4,
                     GL_RED_EXT, GL_UNSIGNED_BYTE, 0);
        tango_gl::util::CheckGlError("glReadPixels");
        end = get_time();
        duration_ns = end - start;

        gm_debug(data->log, "glReadPixels took %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        start = get_time();
        void *pbo_ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0,
                                         ((rotated_frame_width / 4) *
                                          (rotated_frame_height / 4)),
                                         GL_MAP_READ_BIT);
        tango_gl::util::CheckGlError("map buffer range");
        end = get_time();
        duration_ns = end - start;

        gm_debug(data->log, "glMapBufferRange took %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

        {
            dlib::timing::timer lv0_cpy_timer("Copied pyramid level0 frame for face detection from PBO in");

            detect_buf_width_ = rotated_frame_width / 4;
            detect_buf_height_ = rotated_frame_height / 4;

            /* TODO: avoid copying out of the PBO later (assuming we can get a
             * cached mapping)
             */
            gm_debug(data->log, "face detect scratch width = %d, height = %d",
                 (int)detect_buf_width_,
                 (int)detect_buf_height_);
            grey_face_detect_scratch_.resize(detect_buf_width_ * detect_buf_height_);
            memcpy(grey_face_detect_scratch_.data(), pbo_ptr, grey_face_detect_scratch_.size());

            detect_buf_data_ = grey_face_detect_scratch_.data();
            gm_debug(data->log, "detect_buf_data_ = %p", detect_buf_data_);
        }

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        //glDeleteTextures(pyramid_tex_layers_.size(), &pyramid_tex_layers_[0]);
        //glDeleteFramebuffers(pyramid_fbos_.size(), &pyramid_fbos_[0]);

#endif /* DOWNSAMPLE_ON_GPU */

        need_new_scaled_frame_ = false;
        pthread_cond_signal(&scaled_frame_available_cond_);
    }

    // If tracking is lost, further down in this method Scene::Render
    // will not be called. Prevent flickering that would otherwise
    // happen by rendering solid black as a fallback.
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, screen_width_, screen_height_);

    if (!is_scene_camera_configured_)
        InitCameraProjection();

    pthread_mutex_lock(&debug_viz_mutex_);

    if (grey_debug_buffer_.size() && grey_debug_width_) {
        uint64_t start, end, duration_ns;

        glBindTexture(GL_TEXTURE_2D, debug_overlay_->GetTextureId());
        tango_gl::util::CheckGlError("glBindTexture");
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        for (int y = 0; y < 10; y++) {
            uint8_t *p = ((uint8_t *)grey_debug_buffer_.data()) + grey_debug_width_ * y;
            memset(p, 0xff, grey_debug_width_ / 4);
        }
        for (int y = 20; y < (grey_debug_height_ / 4); y++) {
            uint8_t *p = ((uint8_t *)grey_debug_buffer_.data()) + grey_debug_width_ * y;
            memset(p, 0xff, 10);
        }

        //memset(grey_debug_buffer_.data(), 0xff, 3200);
        start = get_time();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                     grey_debug_width_, grey_debug_height_, 0, GL_LUMINANCE,
                     GL_UNSIGNED_BYTE, grey_debug_buffer_.data());
        tango_gl::util::CheckGlError("glTexImage2D");
        end = get_time();
        duration_ns = end - start;
        /*
           gm_debug(data->log, "Uploaded %dx%d luminance debug texture in %.3f%s",
           grey_debug_width_,
           grey_debug_height_,
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));
           */

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    pthread_mutex_unlock(&debug_viz_mutex_);

    // Querying the GPU color image's frame transformation based its timestamp.
    TangoMatrixTransformData matrix_transform;
    TangoSupport_getMatrixTransformAtTime(
        last_gpu_timestamp_, TANGO_COORDINATE_FRAME_AREA_DESCRIPTION,
        TANGO_COORDINATE_FRAME_CAMERA_COLOR, TANGO_SUPPORT_ENGINE_OPENGL,
        TANGO_SUPPORT_ENGINE_OPENGL, display_rotation_, &matrix_transform);
    if (matrix_transform.status_code != TANGO_POSE_VALID) {
        LOGE(
            "GlimpseDemo: Could not find a valid matrix transform at "
            "time %lf for the color camera.",
            last_gpu_timestamp_);
    } else {
        const glm::mat4 area_description_T_color_camera =
            glm::make_mat4(matrix_transform.matrix);
        GLRender(area_description_T_color_camera);
    }
}
#endif

static void
render_ar_video_background(struct glimpse_data *data)
{
    gm_assert(data->log, !!data->ctx, "render_ar_video_background, NULL ctx");

#if 1
    if (data->device_frame != data->visible_frame) {

        gm_assert(data->log, !!data->device_frame, "render_ar_video_background, NULL device_frame");

        struct gm_frame *new_frame = gm_frame_ref(data->device_frame);

        if (data->visible_frame)
            gm_frame_unref(data->visible_frame);
        data->visible_frame = new_frame;

        gm_debug(data->log, "render_ar_video_background DEBUG 0");
        /*
         * Update video from camera
         */
        glBindTexture(GL_TEXTURE_2D, data->gl_vid_tex);

        gm_debug(data->log, "render_ar_video_background DEBUG 1");
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        gm_debug(data->log, "render_ar_video_background DEBUG 2");
        void *video_front = new_frame->video->data;
        gm_debug(data->log, "render_ar_video_background DEBUG 3");
        enum gm_format video_format = data->device_frame->video_format;

        gm_debug(data->log, "render_ar_video_background DEBUG 4");
        switch (video_format) {
        case GM_FORMAT_LUMINANCE_U8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                         data->video_width, data->video_height,
                         0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
            gm_debug(data->log, "render_ar_video_background DEBUG 5");
            break;

        case GM_FORMAT_RGB:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                         data->video_width, data->video_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, video_front);
            gm_debug(data->log, "render_ar_video_background DEBUG 5");
            break;

        case GM_FORMAT_RGBX:
        case GM_FORMAT_RGBA:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                         data->video_width, data->video_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, video_front);
            gm_debug(data->log, "render_ar_video_background DEBUG 5");
            break;

        case GM_FORMAT_UNKNOWN:
        case GM_FORMAT_Z_U16_MM:
        case GM_FORMAT_Z_F32_M:
        case GM_FORMAT_Z_F16_M:
            gm_assert(data->log, 0, "Unexpected format for video buffer");
            break;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
        gm_debug(data->log, "render_ar_video_background DEBUG 6");
    }

    struct gm_tracking *tracking = gm_context_get_latest_tracking(data->ctx);
    if (tracking) {
        gm_tracking_unref(tracking);
    }
#endif

    GLint save_viewport[4];

    glGetIntegerv(GL_VIEWPORT, save_viewport);

    gm_debug(data->log, "render_ar_video_background DEBUG 7");
    //glViewport(0, 0, main_area_size.x/2, main_area_size.y/2);
    glClearColor(1.0, 1.0, 0.0, 0.0);
    gm_debug(data->log, "render_ar_video_background DEBUG 8");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    draw_textured_quad(data, 0, 0, 1, 1);

    gm_debug(data->log, "render_ar_video_background DEBUG 9");
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
