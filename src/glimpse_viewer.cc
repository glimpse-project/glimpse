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
#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_device.h"


#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))
#define LOOP_INDEX(x,y) ((x)[(y) % ARRAY_LEN(x)])

#define TOOLBAR_LEFT_WIDTH 300

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

typedef struct _Data
{
    struct gm_logger *log;
    struct gm_context *ctx;
    struct gm_device *device;

    GLFWwindow *window;
    int win_width;
    int win_height;

    /* A convenience for accessing the depth_camera_intrinsics.width/height */
    int depth_width;
    int depth_height;

    /* A convenience for accessing the video_camera_intrinsics.width/height */
    int video_width;
    int video_height;

    /* A convenience for accessing number of joints in latest tracking */
    int n_joints;
    int n_bones;

    glm::vec3 focal_point;
    float camera_rot_yx[2];
    JSON_Value *joint_map;

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

    JSON_Value *joints_recording_val;
    JSON_Array *joints_recording;
    int requested_recording_len;
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

static GLuint gl_labels_tex;
static GLuint gl_depth_rgb_tex;
static GLuint gl_rgb_tex;
static GLuint gl_vid_tex;

static GLuint gl_db_program;
static GLuint gl_db_attr_depth;
static GLuint gl_db_uni_mvp;
static GLuint gl_db_uni_pt_size;
static GLuint gl_db_uni_depth_size;
static GLuint gl_db_uni_depth_intrinsics;
static GLuint gl_db_uni_video_intrinsics;
static GLuint gl_db_uni_video_size;
static GLuint gl_db_vid_tex;
static GLuint gl_db_depth_bo;

static GLuint gl_cloud_program;
static GLuint gl_cloud_attr_pos;
static GLuint gl_cloud_attr_col;
static GLuint gl_cloud_uni_mvp;
static GLuint gl_cloud_uni_size;
static GLuint gl_joints_bo;
static GLuint gl_bones_bo;
static GLuint gl_cloud_fbo;
static GLuint gl_cloud_depth_bo;
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
draw_properties(struct gm_ui_properties *props)
{
    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        switch (prop->type) {
        case GM_PROPERTY_INT:
            {
                int current_val = gm_prop_get_int(prop), save_val = current_val;
                ImGui::SliderInt(prop->name, &current_val,
                                 prop->int_state.min, prop->int_state.max);
                if (current_val != save_val)
                    gm_prop_set_int(prop, current_val);
            }
            break;
        case GM_PROPERTY_ENUM:
            {
                int current_enumerant = 0, save_enumerant = 0;
                int current_val = gm_prop_get_enum(prop);

                for (int j = 0; j < prop->enum_state.n_enumerants; j++) {
                    if (prop->enum_state.enumerants[j].val == current_val) {
                        current_enumerant = save_enumerant = j;
                        break;
                    }
                }

                std::vector<const char*> labels(prop->enum_state.n_enumerants);
                for (int j = 0; j < prop->enum_state.n_enumerants; j++) {
                    labels[j] = prop->enum_state.enumerants[j].name;
                }

                ImGui::Combo(prop->name, &current_enumerant, labels.data(),
                             labels.size());

                if (current_enumerant != save_enumerant) {
                    int e = current_enumerant;
                    gm_prop_set_enum(prop, prop->enum_state.enumerants[e].val);
                }
            }
            break;
        case GM_PROPERTY_BOOL:
            {
                bool current_val = gm_prop_get_bool(prop),
                     save_val = current_val;
                ImGui::Checkbox(prop->name, &current_val);
                if (current_val != save_val)
                    gm_prop_set_bool(prop, current_val);
            }
            break;
        case GM_PROPERTY_FLOAT:
            {
                float current_val = gm_prop_get_float(prop), save_val = current_val;
                ImGui::SliderFloat(prop->name, &current_val,
                                   prop->float_state.min, prop->float_state.max);
                if (current_val != save_val)
                    gm_prop_set_float(prop, current_val);
            }
            break;
        case GM_PROPERTY_FLOAT_VEC3:
            if (prop->read_only) {
                ImGui::LabelText(prop->name, "%.3f,%.3f,%.3f",
                                 //prop->vec3_state.components[0],
                                 prop->vec3_state.ptr[0],
                                 //prop->vec3_state.components[1],
                                 prop->vec3_state.ptr[1],
                                 //prop->vec3_state.components[2],
                                 prop->vec3_state.ptr[2]);
            } // else TODO
            break;
        }
    }
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

    ImGui::TextDisabled("Device properties...");
    ImGui::Separator();
    ImGui::Spacing();

    struct gm_ui_properties *props = gm_device_get_ui_properties(data->device);
    draw_properties(props);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Mo-Cap properties...");
    ImGui::Separator();
    ImGui::Spacing();

    props = gm_context_get_ui_properties(data->ctx);
    draw_properties(props);

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
    ImGui::Begin("Video Buffer", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_vid_tex, win_size);
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
        int width = main_area_size.x/2;
        int height = main_area_size.y/2;

        // Generate textures
        glBindTexture(GL_TEXTURE_2D, gl_cloud_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        cloud_tex_valid = true;

        // Bind colour/depth to point-cloud fbo
        glBindFramebuffer(GL_FRAMEBUFFER, gl_cloud_fbo);

        glBindRenderbuffer(GL_RENDERBUFFER, gl_cloud_depth_bo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, gl_cloud_depth_bo);

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

    // Calculate the projection matrix
    struct gm_intrinsics *intrinsics =
      gm_device_get_depth_intrinsics(data->device);
    glm::mat4 proj = intrinsics_to_project_matrix(intrinsics, 0.01f, 10);
    glm::mat4 mvp = glm::scale(proj, glm::vec3(1.0, 1.0, -1.0));
    mvp = glm::translate(mvp, data->focal_point);
    mvp = glm::rotate(mvp, data->camera_rot_yx[0], glm::vec3(0.0, 1.0, 0.0));
    mvp = glm::rotate(mvp, data->camera_rot_yx[1], glm::vec3(1.0, 0.0, 0.0));
    mvp = glm::translate(mvp, -data->focal_point);

    // Redraw depth buffer as point-cloud to texture
    glBindFramebuffer(GL_FRAMEBUFFER, gl_cloud_fbo);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, main_area_size.x/2, main_area_size.y/2);

    glUseProgram(gl_db_program);
    glUniformMatrix4fv(gl_db_uni_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(gl_db_uni_pt_size, 2.f);

    glEnableVertexAttribArray(gl_db_attr_depth);
    glBindBuffer(GL_ARRAY_BUFFER, gl_db_depth_bo);
    glVertexAttribPointer(gl_db_attr_depth, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindTexture(GL_TEXTURE_2D, gl_db_vid_tex);
    glDrawArrays(GL_POINTS, 0, data->video_width * data->video_height);

    glDisableVertexAttribArray(gl_db_attr_depth);

    // Redraw joints/bones to texture
    if (data->n_joints) {
        glUseProgram(gl_cloud_program);

        // Set projection transform
        glUniformMatrix4fv(gl_cloud_uni_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

        // Enable vertex arrays for drawing joints/bones
        glEnableVertexAttribArray(gl_cloud_attr_pos);
        glEnableVertexAttribArray(gl_cloud_attr_col);

        // Have bones appear over everything, but depth test them against each
        // other.
        glClear(GL_DEPTH_BUFFER_BIT);

        // Bind bones buffer-object
        glBindBuffer(GL_ARRAY_BUFFER, gl_bones_bo);

        glVertexAttribPointer(gl_cloud_attr_pos, 3, GL_FLOAT,
                              GL_FALSE, sizeof(GlimpsePointXYZRGBA), nullptr);
        glVertexAttribPointer(gl_cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                              GL_TRUE, sizeof(GlimpsePointXYZRGBA),
                              (void *)offsetof(GlimpsePointXYZRGBA, rgba));

        // Draw bone lines
        glDrawArrays(GL_LINES, 0, data->n_bones * 2);

        // Have joint points appear over everything, but depth test them
        // against each other.
        glClear(GL_DEPTH_BUFFER_BIT);

        // Set point size for joints
        glUniform1f(gl_cloud_uni_size, 6.f);

        // Bind joints buffer-object
        glBindBuffer(GL_ARRAY_BUFFER, gl_joints_bo);

        glVertexAttribPointer(gl_cloud_attr_pos, 3, GL_FLOAT,
                              GL_FALSE, sizeof(GlimpsePointXYZRGBA), nullptr);
        glVertexAttribPointer(gl_cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                              GL_TRUE, sizeof(GlimpsePointXYZRGBA),
                              (void *)offsetof(GlimpsePointXYZRGBA, rgba));

        // Draw joint points
        glDrawArrays(GL_POINTS, 0, data->n_joints);

        // Clean-up
        glDisableVertexAttribArray(gl_cloud_attr_pos);
        glDisableVertexAttribArray(gl_cloud_attr_col);
    }

    // Clean-up
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);

    // Draw buffer
    ImGui::Image((void *)(intptr_t)gl_cloud_tex, win_size,
                 ImVec2(0, 0), ImVec2(1, 1));

    if (ImGui::IsWindowHovered()) {
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

    if (data->device_frame) {
        ProfileScopedSection(FreeFrame);
        gm_frame_unref(data->device_frame);
    }

    {
        ProfileScopedSection(GetLatestFrame);
        /* NB: gm_device_get_latest_frame will give us a _ref() */
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
        request_device_frame(data, GM_REQUEST_FRAME_VIDEO);
    }

    if (upload) {
        ProfileScopedSection(UploadFrameTextures);

        /*
         * Update video from camera
         */
        glBindTexture(GL_TEXTURE_2D, gl_vid_tex);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        void *video_front = data->device_frame->video->data;
        enum gm_format video_format = data->device_frame->video_format;

        switch (video_format) {
        case GM_FORMAT_LUMINANCE_U8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                         data->video_width, data->video_height,
                         0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_RGB:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                         data->video_width, data->video_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_RGBX:
        case GM_FORMAT_RGBA:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                         data->video_width, data->video_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_UNKNOWN:
        case GM_FORMAT_Z_U16_MM:
        case GM_FORMAT_Z_F32_M:
        case GM_FORMAT_Z_F16_M:
            gm_assert(data->log, 0, "Unexpected format for video buffer");
            break;
        }
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

    /* Update depth buffer and colour buffer */
    const float *depth = gm_tracking_get_depth(data->latest_tracking);
    glBindBuffer(GL_ARRAY_BUFFER, gl_db_depth_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float) * data->depth_width * data->depth_height,
                 depth, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, gl_db_vid_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    const uint32_t *color = gm_tracking_get_video(data->latest_tracking);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 data->video_width, data->video_height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, color);

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
    data->n_joints = 0;
    const float *joints =
        gm_tracking_get_joint_positions(data->latest_tracking, &data->n_joints);

    // TODO: At some point, the API needs to return a list of joints with ids,
    //       possibly with confidences, so in the situation that a joint isn't
    //       visible (or possible to determine), we can indicate that.
    if (data->n_joints) {
        assert((size_t)data->n_joints ==
               json_array_get_count(json_array(data->joint_map)));
    }

    if (data->n_joints) {
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
                     colored_joints, GL_DYNAMIC_DRAW);

        // Reformat and copy over bone data
        // TODO: Don't parse this JSON structure here
        GlimpsePointXYZRGBA colored_bones[data->n_bones * 2];
        for (int i = 0, b = 0; i < data->n_joints; i++) {
            JSON_Object *joint =
                json_array_get_object(json_array(data->joint_map), i);
            JSON_Array *connections =
                json_object_get_array(joint, "connections");
            for (size_t c = 0; c < json_array_get_count(connections); c++) {
                const char *joint_name = json_array_get_string(connections, c);
                for (int j = 0; j < data->n_joints; j++) {
                    JSON_Object *joint2 = json_array_get_object(
                        json_array(data->joint_map), j);
                    if (strcmp(joint_name,
                               json_object_get_string(joint2, "joint")) == 0) {
                        colored_bones[b++] = colored_joints[i];
                        colored_bones[b++] = colored_joints[j];
                        break;
                    }
                }
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, gl_bones_bo);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(GlimpsePointXYZRGBA) * data->n_bones * 2,
                     colored_bones, GL_DYNAMIC_DRAW);

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

    if (data->latest_tracking)
        gm_tracking_unref(data->latest_tracking);

    data->latest_tracking = gm_context_get_latest_tracking(data->ctx);
    assert(data->latest_tracking);

    if (data->joints_recording) {
        int n_joints;
        const float *joints =
            gm_tracking_get_joint_positions(data->latest_tracking,
                                            &n_joints);
        JSON_Value *joints_array_val = json_value_init_array();
        JSON_Array *joints_array = json_array(joints_array_val);
        for (int i = 0; i < n_joints; i++) {
            const float *joint = joints + 3 * i;
            JSON_Value *coord_val = json_value_init_array();
            JSON_Array *coord = json_array(coord_val);

            json_array_append_number(coord, joint[0]);
            json_array_append_number(coord, joint[1]);
            json_array_append_number(coord, joint[2]);

            json_array_append_value(joints_array, coord_val);
        }

        json_array_append_value(data->joints_recording, joints_array_val);

        int n_frames = json_array_get_count(data->joints_recording);
        if (n_frames >= data->requested_recording_len) {
            json_serialize_to_file_pretty(data->joints_recording_val,
                                          "glimpse-joints-recording.json");
            json_value_free(data->joints_recording_val);
            data->joints_recording_val = NULL;
            data->joints_recording = NULL;
        }
    }

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
                              GM_REQUEST_FRAME_VIDEO));
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

/* NB: it's undefined what thread this is called on so we queue events to
 * be processed as part of the mainloop processing.
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

    static const char *vertShaderDepth =
        "#version 300 es\n"
        "precision mediump float;\n\n"
        "precision mediump int;\n\n"

        "uniform mat4 mvp;\n"
        "uniform float pt_size;\n"
        "uniform ivec2 depth_size;\n"
        "uniform vec4 depth_intrinsics;\n"
        "uniform vec4 video_intrinsics;\n"
        "uniform vec2 video_size;\n\n"

        "in float depth;\n"
        "out vec2 v_tex_coord;\n\n"

        "void main() {\n"
        // Unproject the depth information into 3d space
        "  float fx = depth_intrinsics.x;\n"
        "  float fy = depth_intrinsics.y;\n"
        "  float cx = depth_intrinsics.z;\n"
        "  float cy = depth_intrinsics.w;\n\n"

        "  int x = int(gl_VertexID) % depth_size.x;\n"
        "  int y = int(gl_VertexID) / depth_size.x;\n"
        "  float dx = ((float(x) - cx) * depth) / fx;\n"
        "  float dy = (-(float(y) - cy) * depth) / fy;\n\n"

        // Reproject the depth coordinates into video space
        // TODO: Support extrinsics
        "  fx = video_intrinsics.x;\n"
        "  fy = video_intrinsics.y;\n"
        "  cx = video_intrinsics.z;\n"
        "  cy = video_intrinsics.w;\n"
        "  float tx = ((dx * fx / depth) + cx) / video_size.x;\n"
        "  float ty = ((dy * fy / depth) + cy) / video_size.y;\n"

        // Output values for the fragment shader
        "  gl_PointSize = pt_size;\n"
        "  gl_Position =  mvp * vec4(dx, dy, depth, 1.0);\n"
        "  v_tex_coord = vec2(tx, 1.0 - ty);\n"
        "}\n";

    static const char *fragShaderDepth =
        "#version 300 es\n"
        "precision mediump float;\n"
        "precision mediump int;\n\n"

        "uniform sampler2D texture;\n\n"

        "in vec2 v_tex_coord;\n"
        "layout(location = 0) out vec4 color;\n\n"

        "void main() {\n"
        "  color = texture2D(texture, v_tex_coord.st).abgr;\n"
        "}\n";

    GLuint cloudFragShader, cloudVertShader, depthFragShader, depthVertShader;

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearStencil(0);

    // Create depth-buffer point shader
    depthFragShader = compile_shader(GL_FRAGMENT_SHADER, fragShaderDepth);
    depthVertShader = compile_shader(GL_VERTEX_SHADER, vertShaderDepth);
    gl_db_program = link_program(depthFragShader, depthVertShader, 0);

    glUseProgram(gl_db_program);

    gl_db_attr_depth = glGetAttribLocation(gl_db_program, "depth");
    gl_db_uni_mvp = glGetUniformLocation(gl_db_program, "mvp");
    gl_db_uni_pt_size = glGetUniformLocation(gl_db_program, "pt_size");
    gl_db_uni_depth_size = glGetUniformLocation(gl_db_program, "depth_size");
    gl_db_uni_depth_intrinsics = glGetUniformLocation(gl_db_program,
                                                      "depth_intrinsics");
    gl_db_uni_video_intrinsics = glGetUniformLocation(gl_db_program,
                                                      "video_intrinsics");
    gl_db_uni_video_size = glGetUniformLocation(gl_db_program, "video_size");
    glGenBuffers(1, &gl_db_depth_bo);

    GLuint uniform_tex_sampler = glGetUniformLocation(gl_db_program, "texture");
    glUniform1i(uniform_tex_sampler, 0);

    // Update camera intrinsics
    struct gm_intrinsics *depth_intrinsics =
      gm_device_get_depth_intrinsics(data->device);
    struct gm_intrinsics *video_intrinsics =
      gm_device_get_video_intrinsics(data->device);

    glUniform2i(gl_db_uni_depth_size,
                (GLint)depth_intrinsics->width,
                (GLint)depth_intrinsics->height);
    glUniform2f(gl_db_uni_video_size,
                (GLfloat)video_intrinsics->width,
                (GLfloat)video_intrinsics->height);
    glUniform4f(gl_db_uni_depth_intrinsics,
                (GLfloat)depth_intrinsics->fx,
                (GLfloat)depth_intrinsics->fy,
                (GLfloat)depth_intrinsics->cx,
                (GLfloat)depth_intrinsics->cy);
    glUniform4f(gl_db_uni_video_intrinsics,
                (GLfloat)video_intrinsics->fx,
                (GLfloat)video_intrinsics->fy,
                (GLfloat)video_intrinsics->cx,
                (GLfloat)video_intrinsics->cy);

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
    glGenBuffers(1, &gl_joints_bo);
    glGenBuffers(1, &gl_bones_bo);

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

    glGenTextures(1, &gl_vid_tex);
    glBindTexture(GL_TEXTURE_2D, gl_vid_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_labels_tex);
    glBindTexture(GL_TEXTURE_2D, gl_labels_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_db_vid_tex);
    glBindTexture(GL_TEXTURE_2D, gl_db_vid_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_cloud_tex);
    glBindTexture(GL_TEXTURE_2D, gl_cloud_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &gl_cloud_fbo);
    glGenRenderbuffers(1, &gl_cloud_depth_bo);
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
    if (backtrace) {
        int line_len = 100;
        char *formatted = (char *)alloca(backtrace->n_frames * line_len);

        gm_logger_get_backtrace_strings(logger, backtrace,
                                        line_len, (char *)formatted);
        for (int i = 0; i < backtrace->n_frames; i++) {
            char *line = formatted + line_len * i;
            fprintf(stderr, "> %s\n", line);
        }
    }
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

    const char *font_ttf = "Roboto-Medium.ttf";
    char font_asset_path[512];
    const char *font_path = NULL;

    const char *joint_map_json = "joint-map.json";
    char joint_map_asset_path[512];
    const char *joint_map_path = NULL;

    char *assets_root = getenv("GLIMPSE_ASSETS_ROOT");
    if (assets_root) {
        xsnprintf(font_asset_path, sizeof(font_asset_path), "%s/%s",
                  assets_root, font_ttf);
        font_path = font_asset_path;
        xsnprintf(joint_map_asset_path, sizeof(joint_map_asset_path), "%s/%s",
                  assets_root, joint_map_json);
        joint_map_path = joint_map_asset_path;
    } else {
        font_path = font_ttf;
        joint_map_path = joint_map_json;
    }

    io.Fonts->AddFontFromFileTTF(font_path, 16.0f);

    const char *n_frames_env = getenv("GLIMPSE_RECORD_N_JOINT_FRAMES");
    if (n_frames_env) {
        data.joints_recording_val = json_value_init_array();
        data.joints_recording = json_array(data.joints_recording_val);
        data.requested_recording_len = strtoull(n_frames_env, NULL, 10);
    }

    // TODO: Might be nice to be able to retrieve this information via the API
    //       rather than reading it separately here.
    data.joint_map = json_parse_file(joint_map_path);

    // Count the number of bones defined by connections in the joint map.
    data.n_bones = 0;
    for (size_t i = 0; i < json_array_get_count(json_array(data.joint_map));
         i++) {
        JSON_Object *joint =
            json_array_get_object(json_array(data.joint_map), i);
        data.n_bones += json_array_get_count(
            json_object_get_array(joint, "connections"));
    }

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    init_opengl(&data);

    data.ctx = gm_context_new(data.log, NULL);

    gm_context_set_depth_camera_intrinsics(data.ctx, depth_intrinsics);
    gm_context_set_video_camera_intrinsics(data.ctx, video_intrinsics);
    /*gm_context_set_depth_to_video_camera_extrinsics(data.ctx,
        gm_device_get_depth_to_video_extrinsics(data.device));*/

    /* NB: there's no guarantee about what thread these event callbacks
     * might be invoked from...
     */
    gm_context_set_event_callback(data.ctx, on_event_cb, &data);
    gm_device_set_event_callback(data.device, on_device_event_cb, &data);

    gm_device_start(data.device);
    gm_context_enable(data.ctx);

    event_loop(&data);

    /* Destroying the context' tracking pool will assert that all tracking
     * resources have been released first...
     */
    if (data.latest_tracking)
        gm_tracking_unref(data.latest_tracking);

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

    if (data.device_frame)
        gm_frame_unref(data.device_frame);

    gm_device_close(data.device);

    ProfileShutdown();
    ImGui_ImplGlfwGLES3_Shutdown();
    glfwDestroyWindow(data.window);
    glfwTerminate();

    json_value_free(data.joint_map);

    gm_logger_destroy(data.log);

    delete data.events_front;
    delete data.events_back;

    return 0;
}
