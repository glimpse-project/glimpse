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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <locale.h>
#include <assert.h>
#include <time.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <math.h>

#include <pthread.h>

#include <vector>

#include <epoxy/gl.h>
#include <epoxy/egl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw_gles3.h>
#include <profiler.h>

#include "half.hpp"

#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_device.h"


#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))
#define LOOP_INDEX(x,y) ((x)[(y) % ARRAY_LEN(x)])

#define TOOLBAR_LEFT_WIDTH 300

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

typedef struct _Data
{
    struct gm_logger *log;
    struct gm_context *ctx;
    struct gm_device *device;

    GLFWwindow *window;
    int win_width;
    int win_height;

    int attr_pos;
    int attr_tex_coord;
    int attr_color;

    /* A convenience for accessing the depth_camera_intrinsics.width/height */
    int depth_width;
    int depth_height;

    /* A convenience for accessing the video_camera_intrinsics.width/height */
    int video_width;
    int video_height;

    /* A convenience for accessing number of points/joints in latest tracking */
    int n_points;
    int n_joints;

    glm::vec3 focal_point;
    float camera_rot_yx[2];

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
} Data;

static uint32_t joint_palette[] = {
    0xFFFFFFFF, // head.tail
    0xCCCCCCCC, // neck_01.head
    0xFF8888FF, // upperarm_l.head
    0x8888FFFF, // upperarm_r.head
    0xFFFF88FF, // lowerarm_l.head
    0xFFFF00FF, // lowerarm_l.tail
    0x88FFFFFF, // lowerarm_r.head
    0x00FFFFFF, // lowerarm_r.tail
    0x33FF33FF, // thigh_l.head
    0x33AA33FF, // thigh_l.tail
    0xFFFF33FF, // thigh_r.head
    0xAAAA33FF, // thigh_r.tail
    0x3333FFFF, // foot_l.head
    0xFF3333FF, // foot_r.head
};

static GLuint gl_tex_program;
static GLuint uniform_tex_sampler;
static GLuint gl_labels_tex;
static GLuint gl_depth_rgb_tex;
static GLuint gl_rgb_tex;
static GLuint gl_lum_tex;

static GLuint gl_cloud_program;
static GLuint gl_cloud_attr_pos;
static GLuint gl_cloud_attr_col;
static GLuint gl_cloud_uni_mvp;
static GLuint gl_cloud_uni_size;
static GLuint gl_cloud_bo;
static GLuint gl_joints_bo;
static GLuint gl_cloud_fbo;
static GLuint gl_cloud_tex;
static bool cloud_tex_valid = false;

static bool pause_profile;

static void
on_profiler_pause_cb(bool pause)
{
    pause_profile = pause;
}

glm::mat4
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

static void
draw_ui(Data *data)
{
    float left_col = TOOLBAR_LEFT_WIDTH;
    ImVec2 main_menu_size;
    ImVec2 win_size;
    ProfileScopedSection(DrawIMGUI, ImGuiControl::Profiler::Dark);

    ImGui_ImplGlfwGLES3_NewFrame();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

    if (ImGui::BeginMainMenuBar()) {
        main_menu_size = ImGui::GetWindowSize();

        if (ImGui::BeginMenu("File")) {

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::SetNextWindowPos(ImVec2(0, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(left_col, data->win_height));
    ImGui::Begin("Controls", NULL,
                 ImGuiWindowFlags_NoTitleBar|
                 ImGuiWindowFlags_NoResize);

    struct gm_ui_properties *props = gm_context_get_ui_properties(data->ctx);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        if (prop->type == GM_PROPERTY_INT)
            ImGui::SliderInt(prop->name, prop->int_ptr, prop->min, prop->max);
        if (prop->type == GM_PROPERTY_FLOAT)
            ImGui::SliderFloat(prop->name, prop->float_ptr, prop->min, prop->max);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    props = gm_device_get_ui_properties(data->device);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        switch (prop->type) {
        case GM_PROPERTY_INT:
            ImGui::SliderInt(prop->name, prop->int_ptr, prop->min, prop->max);
            break;
        case GM_PROPERTY_FLOAT:
            ImGui::SliderFloat(prop->name, prop->float_ptr, prop->min, prop->max);
            break;
        case GM_PROPERTY_FLOAT_VEC3:
            if (prop->read_only) {
                ImGui::LabelText(prop->name, "%.3f,%.3f,%.3f",
                                 prop->float_vec3[0],
                                 prop->float_vec3[1],
                                 prop->float_vec3[2]);
            } // else TODO
            break;
        }
    }

    ImGui::End();

    ImVec2 main_area_size = ImVec2(data->win_width - left_col, data->win_height - main_menu_size.y);

    ImGui::SetNextWindowPos(ImVec2(left_col, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Depth Buffer", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_depth_rgb_tex, win_size);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(left_col, main_menu_size.y + main_area_size.y/2));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Luminance", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_lum_tex, win_size);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(left_col + main_area_size.x/2, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Labels", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_labels_tex, win_size);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(left_col + main_area_size.x/2,
                                   main_menu_size.y + main_area_size.y/2));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Cloud", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();

    // Ensure the framebuffer texture is valid
    if (!cloud_tex_valid) {
        glBindTexture(GL_TEXTURE_2D, gl_cloud_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     main_area_size.x/2, main_area_size.y/2,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        cloud_tex_valid = true;

        glBindFramebuffer(GL_FRAMEBUFFER, gl_cloud_fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             gl_cloud_tex, 0);
        GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
           GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "Incomplete framebuffer\n");
            exit(1);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Redraw point-clouds to texture
    glBindFramebuffer(GL_FRAMEBUFFER, gl_cloud_fbo);

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, main_area_size.x/2, main_area_size.y/2);

    // Enable point-cloud drawing shader
    glUseProgram(gl_cloud_program);

    // Set projection transform
    struct gm_intrinsics *intrinsics =
      gm_device_get_depth_intrinsics(data->device);
    glm::mat4 proj = intrinsics_to_project_matrix(intrinsics, 0.01f, 10);
    glm::mat4 mvp = glm::scale(proj, glm::vec3(1.0, 1.0, -1.0));
    mvp = glm::translate(mvp, data->focal_point);
    mvp = glm::rotate(mvp, data->camera_rot_yx[0], glm::vec3(0.0, 1.0, 0.0));
    mvp = glm::rotate(mvp, data->camera_rot_yx[1], glm::vec3(1.0, 0.0, 0.0));
    mvp = glm::translate(mvp, -data->focal_point);

    glUniformMatrix4fv(gl_cloud_uni_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    // Enable vertex arrays for drawing point-clouds
    glEnableVertexAttribArray(gl_cloud_attr_pos);
    glEnableVertexAttribArray(gl_cloud_attr_col);

    if (data->n_points) {
        // Set point size
        glUniform1f(gl_cloud_uni_size, 1.f);

        // Bind point cloud buffer-object
        glBindBuffer(GL_ARRAY_BUFFER, gl_cloud_bo);

        glVertexAttribPointer(gl_cloud_attr_pos, 3, GL_FLOAT,
                              GL_FALSE, // normalized
                              sizeof(GlimpsePointXYZRGBA), // stride
                              nullptr); // bo offset
        glVertexAttribPointer(gl_cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                              GL_TRUE,
                              sizeof(GlimpsePointXYZRGBA),
                              (void *)offsetof(GlimpsePointXYZRGBA, rgba));

        // Draw labelled point cloud
        glDrawArrays(GL_POINTS, 0, data->n_points);
    }

    if (data->n_joints) {
        // Set point size for joints
        glUniform1f(gl_cloud_uni_size, 6.f);

        // Bind joints buffer-object
        glBindBuffer(GL_ARRAY_BUFFER, gl_joints_bo);

        glVertexAttribPointer(gl_cloud_attr_pos, 3, GL_FLOAT,
                              GL_FALSE, // normalized
                              sizeof(GlimpsePointXYZRGBA), // stride
                              nullptr); // bo offset
        glVertexAttribPointer(gl_cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                              GL_TRUE,
                              sizeof(GlimpsePointXYZRGBA),
                              (void *)offsetof(GlimpsePointXYZRGBA, rgba));

        // Draw joint points
        glDrawArrays(GL_POINTS, 0, data->n_joints);
    }

    // Clean-up
    glDisableVertexAttribArray(gl_cloud_attr_pos);
    glDisableVertexAttribArray(gl_cloud_attr_col);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    ImGui::ImageButton((void *)(intptr_t)gl_cloud_tex, win_size,
                       ImVec2(0, 0), ImVec2(1, 1), 0);

    if (ImGui::IsItemActive()) {
        if (ImGui::IsMouseDragging()) {
            ImVec2 drag_delta = ImGui::GetMouseDragDelta();
            data->camera_rot_yx[0] += (drag_delta.x * M_PI / 180.f) * 0.2f;
            data->camera_rot_yx[1] += (drag_delta.y * M_PI / 180.f) * 0.2f;
            ImGui::ResetMouseDragDelta();
        }
    }

    ImGui::End();

    ImGui::PopStyleVar();

    /* Handy to look at the various features of imgui... */
#if 0
    ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
    ImGui::ShowTestWindow(&show_test_window);
#endif

    ProfileDrawUI();

    ImGui::Render();
}

static void
redraw(Data *data)
{
    ProfileScopedSection(Redraw);

    glViewport(0, 0, data->win_width, data->win_height);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_ui(data);
}

/* If we've already requested gm_device for a frame then this won't submit
 * a request that downgrades the requirements
 */
static void
request_device_frame(Data *data, uint64_t requirements)
{
    uint64_t new_requirements = data->pending_frame_requirements | requirements;

    if (data->pending_frame_requirements != new_requirements) {
        gm_device_request_frame(data->device, new_requirements);
        data->pending_frame_requirements = new_requirements;
    }
}

static void
handle_device_frame_updates(Data *data)
{
    ProfileScopedSection(UpdatingDeviceFrame);
    bool upload = false;

    if (!data->device_frame_ready)
        return;

    /* XXX We have to consider that a gm_frame currently only remains valid
     * to access until the next call to gm_device_get_latest_frame().
     *
     * Conceptually we have two decoupled consumers: 1) this redraw/render
     * loop 2) skeletal tracking so we need to be careful about
     * understanding the required gm_frame lifetime.
     *
     * Since we can currently assume gm_context_notify_frame() will
     * internally copy whatever frame data it requires then so long as we
     * synchronize these calls with the redraw loop we know it's safe to
     * free the last gm_frame once we have received a new one.
     */

    if (data->device_frame) {
        ProfileScopedSection(FreeFrame);
        gm_device_free_frame(data->device, data->device_frame);
    }

    {
        ProfileScopedSection(GetLatestFrame);
        data->device_frame = gm_device_get_latest_frame(data->device);
        assert(data->device_frame);
        upload = true;
    }

    if (data->context_needs_frame) {
        ProfileScopedSection(FwdContextFrame);

        data->context_needs_frame =
            !gm_context_notify_frame(data->ctx, data->device_frame);
    }

    data->device_frame_ready = false;

    {
        ProfileScopedSection(DeviceFrameRequest);

        /* immediately request a new frame since we want to render the camera
         * at the native capture rate, even though we might not be tracking
         * at that rate.
         *
         * Note: the requirements may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, GM_REQUEST_FRAME_LUMINANCE);
    }

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
}

static void
upload_tracking_textures(Data *data)
{
    ProfileScopedSection(UploadTrackingBufs);

    /*
     * Update the RGB visualization of the depth buffer
     */
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    const uint8_t *depth_rgb = gm_tracking_get_rgb_depth(data->latest_tracking);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->depth_width, data->depth_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, depth_rgb);

    /*
     * Update inferred label map
     */
    glBindTexture(GL_TEXTURE_2D, gl_labels_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    int label_map_width = 0;
    int label_map_height = 0;

    const uint8_t *labels_rgb =
        gm_tracking_get_rgb_label_map(data->latest_tracking,
                                      &label_map_width,
                                      &label_map_height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 label_map_width, label_map_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, labels_rgb);

    /*
     * Update labelled point cloud
     */
    data->n_points = 0;
    data->n_joints = 0;
    const GlimpsePointXYZRGBA *cloud =
        gm_tracking_get_rgb_label_cloud(data->latest_tracking, &data->n_points);
    const float *joints =
        gm_tracking_get_joint_positions(data->latest_tracking, &data->n_joints);

    if (data->n_points) {
        // Copy point cloud data to GPU
        glBindBuffer(GL_ARRAY_BUFFER, gl_cloud_bo);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(GlimpsePointXYZRGBA) * data->n_points,
                     &cloud[0], GL_DYNAMIC_DRAW);

        // Reformat and copy over joint data
        GlimpsePointXYZRGBA colored_joints[data->n_joints];
        for (int i = 0, off = 0; i < data->n_joints; i++) {
            colored_joints[i].x = joints[off++];
            colored_joints[i].y = joints[off++];
            colored_joints[i].z = joints[off++];
            colored_joints[i].rgba = LOOP_INDEX(joint_palette, i);
        }
        glBindBuffer(GL_ARRAY_BUFFER, gl_joints_bo);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(GlimpsePointXYZRGBA) * data->n_joints,
                     &colored_joints[0], GL_DYNAMIC_DRAW);

        // Clean-up
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

static void
handle_context_tracking_updates(Data *data)
{
    ProfileScopedSection(UpdatingTracking);

    if (!data->tracking_ready)
        return;

    data->tracking_ready = false;
    data->latest_tracking = gm_context_get_latest_tracking(data->ctx);
    assert(data->latest_tracking);

    upload_tracking_textures(data);
}

static void
handle_device_event(Data *data, struct gm_device_event *event)
{
    switch (event->type) {
    case GM_DEV_EVENT_FRAME_READY:

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
handle_context_event(Data *data, struct gm_event *event)
{
    switch (event->type) {
    case GM_EVENT_REQUEST_FRAME:
        data->context_needs_frame = true;
        request_device_frame(data,
                             (GM_REQUEST_FRAME_DEPTH |
                              GM_REQUEST_FRAME_LUMINANCE));
        break;
    case GM_EVENT_TRACKING_READY:
        data->tracking_ready = true;
        break;
    }

    gm_context_event_free(event);
}

static void
event_loop(Data *data)
{
    while (!glfwWindowShouldClose(data->window)) {
        ProfileNewFrame();

        ProfileScopedSection(Frame);

        {
            ProfileScopedSection(GLFWEvents);
            glfwPollEvents();
        }

        {
            ProfileScopedSection(GlimpseEvents);

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
        }

        handle_device_frame_updates(data);
        handle_context_tracking_updates(data);

        {
            ProfileScopedSection(GlimpseGPUHook);
            gm_context_render_thread_hook(data->ctx);
        }

        redraw(data);

        {
            ProfileScopedSection(SwapBuffers);
            glfwSwapBuffers(data->window);
        }
    }
}

static void
on_window_fb_size_change_cb(GLFWwindow *window, int width, int height)
{
    Data *data = (Data *)glfwGetWindowUserPointer(window);

    data->win_width = width;
    data->win_height = height;
    cloud_tex_valid = false;
}

static void
on_key_input_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Data *data = (Data *)glfwGetWindowUserPointer(window);

    if (action != GLFW_PRESS)
        return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(data->window, 1);
        break;
    }

    ImGui_ImplGlfwGLES3_KeyCallback(window, key, scancode, action, mods);
}

static void
on_glfw_error_cb(int error_code, const char *error_msg)
{
    fprintf(stderr, "GLFW ERROR: %d: %s\n", error_code, error_msg);
}

static void
on_khr_debug_message_cb(GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum gl_severity,
                        GLsizei length,
                        const GLchar *message,
                        void *userParam)
{
    switch (gl_severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        fprintf(stderr, "GL DEBUG: HIGH SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        fprintf(stderr, "GL DEBUG: MEDIUM SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        fprintf(stderr, "GL DEBUG: LOW SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        fprintf(stderr, "GL DEBUG: NOTIFICATION: %s\n", message);
        break;
    }
}

/* NB: it's undefined what thread this is called on and we are currently
 * assuming it's safe to call gm_device_request_frame() from any thread
 * considering that it just sets a bitmask and signals a condition variable.
 */
static void
on_event_cb(struct gm_context *ctx,
            struct gm_event *context_event, void *user_data)
{
    Data *data = (Data *)user_data;

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
    Data *data = (Data *)user_data;

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
}

static GLuint
compile_shader(GLenum shaderType, const char *shaderText)
{
    GLint stat;
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const char **) &shaderText, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &stat);
    if (!stat) {
        char log[1000];
        GLsizei len;
        glGetShaderInfoLog (shader, 1000, &len, log);
        fprintf(stderr, "Error: Shader did not compile: %s\n", log);
        exit(1);
    }

    return shader;
}

static GLuint
link_program(GLuint firstShader, ...)
{
    GLint stat;
    GLuint program = glCreateProgram();

    glAttachShader(program, firstShader);

    va_list args;
    va_start(args, firstShader);

    GLuint shader;
    while ((shader = va_arg(args, GLuint))) {
        glAttachShader(program, shader);
    }
    va_end(args);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &stat);
    if (!stat) {
        char log[1000];
        GLsizei len;
        glGetProgramInfoLog(program, 1000, &len, log);
        fprintf (stderr, "Error linking:\n%s\n", log);
        exit (1);
    }

    return program;
}

static void
init_opengl(Data *data)
{
    static const char *vertShaderText =
        "#version 300 es\n"
        "precision mediump float;\n"
        "precision mediump int;\n"
        "in vec4 pos;\n"
        "in vec4 color;\n"
        "in vec4 tex_coord;\n"
        "out vec4 v_tex_coord;\n"
        "out vec4 v_color;\n"
        "\n"
        "void main () {\n"
        "  gl_Position = pos;\n"
        "  v_tex_coord = tex_coord;\n"
        "  v_color = color;\n"
        "}\n";

    static const char *fragShaderText =
        "#version 300 es\n"
        "precision mediump float;\n"
        "precision mediump int;\n"
        "in vec4 v_tex_coord;\n"
        "in vec4 v_color;\n"
        "out vec4 frag_color;\n"
        "uniform sampler2D texture;\n"
        "\n"
        "void main () {\n"
        "  frag_color = texture2D(texture, v_tex_coord.st) * v_color;\n"
        "}\n";

    static const char *vertShaderCloud =
        "#version 300 es\n"
        "precision mediump float;\n"
        "uniform mat4 mvp;\n"
        "uniform float size;\n"
        "in vec3 pos;\n"
        "in vec4 color_in;\n"
        "out vec4 v_color;\n"
        "\n"
        "void main() {\n"
        "  gl_PointSize = size;\n"
        "  gl_Position =  mvp * vec4(pos, 1.0);\n"
        "  v_color = color_in;\n"
        "}\n";

    static const char *fragShaderCloud =
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec4 v_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  color = v_color.abgr;\n"
        "}\n";

    GLuint textFragShader, textVertShader, cloudFragShader, cloudVertShader;

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClearStencil(0);

    // Create texture shader
    textFragShader = compile_shader(GL_FRAGMENT_SHADER, fragShaderText);
    textVertShader = compile_shader(GL_VERTEX_SHADER, vertShaderText);
    gl_tex_program = link_program(textFragShader, textVertShader, 0);

    glUseProgram(gl_tex_program);

    uniform_tex_sampler = glGetUniformLocation(gl_tex_program, "texture");
    glUniform1i(uniform_tex_sampler, 0);

    glBindAttribLocation(gl_tex_program, 0, "pos");
    data->attr_pos = 0;

    //data->attr_pos = glGetAttribLocation (gl_tex_program, "pos");
    data->attr_tex_coord = glGetAttribLocation(gl_tex_program, "tex_coord");
    data->attr_color = glGetAttribLocation(gl_tex_program, "color");

    glUseProgram(0);

    // Create point-cloud shader
    cloudFragShader = compile_shader(GL_FRAGMENT_SHADER, fragShaderCloud);
    cloudVertShader = compile_shader(GL_VERTEX_SHADER, vertShaderCloud);
    gl_cloud_program = link_program(cloudFragShader, cloudVertShader, 0);

    glUseProgram(gl_cloud_program);

    gl_cloud_attr_pos = glGetAttribLocation(gl_cloud_program, "pos");
    gl_cloud_attr_col = glGetAttribLocation(gl_cloud_program, "color_in");
    gl_cloud_uni_mvp = glGetUniformLocation(gl_cloud_program, "mvp");
    gl_cloud_uni_size = glGetUniformLocation(gl_cloud_program, "size");
    glGenBuffers(1, &gl_cloud_bo);
    glGenBuffers(1, &gl_joints_bo);

    glUseProgram(0);

    // Generate texture objects
    glGenTextures(1, &gl_depth_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_depth_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_lum_tex);
    glBindTexture(GL_TEXTURE_2D, gl_lum_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_labels_tex);
    glBindTexture(GL_TEXTURE_2D, gl_labels_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_cloud_tex);
    glBindTexture(GL_TEXTURE_2D, gl_cloud_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &gl_cloud_fbo);
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
    //Data *data = (Data *)user_data;

    switch (level) {
    case GM_LOG_ERROR:
        fprintf(stderr, "%s: ERROR: ", context);
        break;
    case GM_LOG_WARN:
        fprintf(stderr, "%s: WARN: ", context);
        break;
    default:
        fprintf(stderr, "%s: ", context);
    }

    vfprintf(stderr, format, ap);
    fprintf(stderr, "\n");
}

int
main(int argc, char **argv)
{
    Data data = {};

    data.log = gm_logger_new(logger_cb, &data);

    data.events_front = new std::vector<struct event>();
    data.events_back = new std::vector<struct event>();
    data.focal_point = glm::vec3(0.0, 0.0, 2.5);

    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW, OpenGL windows system library\n");
        exit(1);
    }

    if (argc == 2) {
        struct gm_device_config config = {};
        config.type = GM_DEVICE_RECORDING;
        config.recording.path = argv[1];
        data.device = gm_device_open(data.log, &config, NULL);
    } else {
        struct gm_device_config config = {};
        config.type = GM_DEVICE_KINECT;
        data.device = gm_device_open(data.log, &config, NULL);
    }

    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(data.device);
    data.depth_width = depth_intrinsics->width;
    data.depth_height = depth_intrinsics->height;

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(data.device);
    data.video_width = video_intrinsics->width;
    data.video_height = video_intrinsics->height;

    data.win_width = 800 + TOOLBAR_LEFT_WIDTH;
    data.win_height = 600;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  0) ;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);

    data.window = glfwCreateWindow(data.win_width,
                                   data.win_height,
                                   "Glimpse Viewer", NULL, NULL);
    if (!data.window) {
        fprintf(stderr, "Failed to create window\n");
        exit(1);
    }

    glfwSetWindowUserPointer(data.window, &data);

    glfwSetFramebufferSizeCallback(data.window, on_window_fb_size_change_cb);

    glfwMakeContextCurrent(data.window);
    glfwSwapInterval(1);

    glfwSetErrorCallback(on_glfw_error_cb);


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
    glDebugMessageCallback((GLDEBUGPROC)on_khr_debug_message_cb, &data);

    ImGui_ImplGlfwGLES3_Init(data.window, false /* don't install callbacks */);

    /* will chain on to ImGui_ImplGlfwGLES3_KeyCallback... */
    glfwSetKeyCallback(data.window, on_key_input_cb);
    glfwSetMouseButtonCallback(data.window, ImGui_ImplGlfwGLES3_MouseButtonCallback);
    glfwSetScrollCallback(data.window, ImGui_ImplGlfwGLES3_ScrollCallback);
    glfwSetCharCallback(data.window, ImGui_ImplGlfwGLES3_CharCallback);

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", 16.0f);

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    init_opengl(&data);

    data.ctx = gm_context_new(data.log, NULL);

    gm_context_set_depth_camera_intrinsics(data.ctx, depth_intrinsics);
    gm_context_set_video_camera_intrinsics(data.ctx, video_intrinsics);

    /* NB: there's no guarantee about what thread these event callbacks
     * might be invoked from...
     */
    gm_context_set_event_callback(data.ctx, on_event_cb, &data);
    gm_device_set_event_callback(data.device, on_device_event_cb, &data);

    gm_device_start(data.device);
    gm_context_enable(data.ctx);

    event_loop(&data);

    gm_context_destroy(data.ctx);

    ProfileShutdown();
    ImGui_ImplGlfwGLES3_Shutdown();
    glfwDestroyWindow(data.window);
    glfwTerminate();

    gm_device_close(data.device);

    delete data.events_front;
    delete data.events_back;

    return 0;
}
