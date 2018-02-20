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

#include <list>
#include <vector>

#include <epoxy/gl.h>
#include <epoxy/egl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_internal.h> // For PushItemFlags(ImGuiItemFlags_Disabled)

#ifdef __ANDROID__
#    include <android/log.h>
#    include <jni.h>
#    include <glfm.h>
#    include <imgui_impl_glfm_gles3.h>
#else
#    define GLFW_INCLUDE_NONE
#    include <GLFW/glfw3.h>
#    include <imgui_impl_glfw_gles3.h>
#endif

#include <profiler.h>

#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_device.h"
#include "glimpse_record.h"
#include "glimpse_assets.h"
#include "glimpse_gl.h"


#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))
#define LOOP_INDEX(x,y) ((x)[(y) % ARRAY_LEN(x)])

#define TOOLBAR_WIDTH 300
#define MAX_VIEWS 4

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)

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
        int android_event;
    };
};

typedef struct _Data
{
    struct gm_logger *log;
    FILE *log_fp;

    /* On Android we don't actually initialize a lot of state including
     * ImGui until we've negotiated permissions, since we might not be
     * able to load the font we need. viewer_init() will be called if
     * the check passes.
     */
    bool initialized;
    bool gl_initialized;

    struct gm_context *ctx;

#ifdef USE_GLFW
    GLFWwindow *window;
#else
    bool surface_created;
#endif
    int win_width;
    int win_height;

    /* The size of the depth buffer visualisation texture */
    int depth_rgb_width;
    int depth_rgb_height;

    /* The size of the video buffer visualisation texture */
    int video_rgb_width;
    int video_rgb_height;

    /* The size of the normals visualisation texture */
    int normals_rgb_width;
    int normals_rgb_height;

    /* The size of the normal clusters visualisation texture */
    int nclusters_rgb_width;
    int nclusters_rgb_height;

    /* The size of the candidate clusters visualisation texture */
    int cclusters_rgb_width;
    int cclusters_rgb_height;

    /* The size of the labels visualisation texture */
    int labels_rgb_width;
    int labels_rgb_height;

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
     * we store the latest frames from gm_device_get_latest_frame() here...
     */
    struct gm_frame *last_depth_frame;
    struct gm_frame *last_video_frame;

    /* Set when gm_context sends a _REQUEST_FRAME event */
    bool context_needs_frame;
    /* Set when gm_context sends a _TRACKING_READY event */
    bool tracking_ready;

    /* Once we've been notified that there's a skeleton tracking update for us
     * then we store the latest tracking data from
     * gm_context_get_latest_tracking() here...
     */
    struct gm_tracking *latest_tracking;

    /* Recording is handled by the gm_recording structure, which saves out
     * frames as we add them.
     */
    bool overwrite_recording;
    struct gm_recording *recording;
    struct gm_device *recording_device;

    struct gm_device *playback_device;

    struct gm_device *active_device;

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

#ifdef __ANDROID__
static JavaVM *android_jvm_singleton;
#endif

static uint32_t joint_palette[] = {
    0xFFFFFFFF, // head.tail
    0xCCCCCCFF, // neck_01.head
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
static GLuint gl_normals_rgb_tex;
static GLuint gl_nclusters_rgb_tex;
static GLuint gl_cclusters_rgb_tex;
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

static const char *views[] = {
    "Controls", "Video Buffer", "Depth Buffer",
    "Normals", "Normal clusters", "Candidate clusters", "Labels", "Cloud" };

static bool cloud_tex_valid = false;

static bool pause_profile;

#ifdef USE_GLFM
static bool permissions_check_failed;
static bool permissions_check_passed;
#endif

static void viewer_init(Data *data);

static void init_viewer_opengl(Data *data);
static void init_basic_opengl(Data *data);
static void handle_device_ready(Data *data, struct gm_device *dev);
static void on_device_event_cb(struct gm_device_event *device_event,
                               void *user_data);

static void
unref_device_frames(Data *data)
{
    if (data->last_video_frame) {
        gm_frame_unref(data->last_video_frame);
        data->last_video_frame = NULL;
    }
    if (data->last_depth_frame) {
        gm_frame_unref(data->last_depth_frame);
        data->last_depth_frame = NULL;
    }
}

static void
on_profiler_pause_cb(bool pause)
{
    pause_profile = pause;
}

glm::mat4
intrinsics_to_project_matrix(const struct gm_intrinsics *intrinsics,
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
adjust_aspect(ImVec2 &input, int width, int height)
{
    ImVec2 output = input;
    float aspect = width / (float)height;
    if (aspect > input.x / input.y) {
        output.y = input.x / aspect;
    } else {
        output.x = input.y * aspect;
    }
    ImVec2 cur = ImGui::GetCursorPos();
    ImGui::SetCursorPosX(cur.x + (input.x - output.x) / 2.f);
    ImGui::SetCursorPosY(cur.y + (input.y - output.y) / 2.f);
    input = output;
}

static bool
draw_controls(Data *data, int x, int y, int width, int height, bool disabled)
{
    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    ImGui::Begin("Controls", NULL,
                 ImGuiWindowFlags_NoTitleBar|
                 ImGuiWindowFlags_NoResize|
                 ImGuiWindowFlags_NoMove|
                 ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (disabled) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    }

    bool focused = ImGui::IsWindowFocused();

    ImGui::TextDisabled("Viewer properties...");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Checkbox("Overwrite recording", &data->overwrite_recording);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Device properties...");
    ImGui::Separator();
    ImGui::Spacing();

    struct gm_ui_properties *props =
        gm_device_get_ui_properties(data->active_device);
    draw_properties(props);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Mo-Cap properties...");
    ImGui::Separator();
    ImGui::Spacing();

    props = gm_context_get_ui_properties(data->ctx);
    draw_properties(props);

    ImGui::Spacing();
    ImGui::Separator();

    if (ImGui::Button("Save config")) {
        char *json = gm_config_save(data->log, props);
        const char *assets_root = getenv("GLIMPSE_ASSETS_ROOT");
        if (!assets_root)
            assets_root = ".";
        char filename[512];

        if (snprintf(filename, sizeof(filename), "%s/%s",
                     assets_root, "glimpse-config.json") <
            (int)sizeof(filename))
        {
            FILE *output = fopen(filename, "w");
            if (output) {
                if (fputs(json, output) == EOF) {
                    gm_error(data->log, "Error writing config: %s",
                             strerror(errno));
                } else {
                    gm_debug(data->log, "Wrote %s", filename);
                }
                if (fclose(output) == EOF) {
                    gm_error(data->log, "Error closing config: %s",
                             strerror(errno));
                }
            } else {
                gm_error(data->log, "Error saving config: %s", strerror(errno));
            }
        }

        free(json);
    }

    if (disabled) {
        ImGui::PopItemFlag();
    }

    ImGui::End();

    return focused;
}

static void
viewer_close_playback_device(Data *data)
{
    gm_device_stop(data->playback_device);

    unref_device_frames(data);

    if (data->latest_tracking) {
        gm_tracking_unref(data->latest_tracking);
        data->latest_tracking = nullptr;
    }

    // Flush old device-dependent data from the context
    gm_context_flush(data->ctx, NULL);
    data->tracking_ready = false;

    gm_device_close(data->playback_device);
    data->playback_device = nullptr;

    data->active_device = data->recording_device;
}

static void
draw_playback_controls(Data *data, const ImVec4 &bounds)
{
    ImGui::Begin("Playback controls", NULL,
                 ImGuiWindowFlags_NoTitleBar|
                 ImGuiWindowFlags_NoResize|
                 ImGuiWindowFlags_ShowBorders|
                 ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::Spacing();

#if 0
    // TODO: Playback controls
    ImGui::Button("<");
    ImGui::SameLine();
    ImGui::Button("||");
    ImGui::SameLine();
    ImGui::Button(">");
    ImGui::SameLine();
#endif
    if (ImGui::Button(data->recording ? "Stop" : "Record")) {
        if (data->recording) {
            gm_recording_close(data->recording);
            data->recording = NULL;
        } else if (!data->playback_device) {
            const char *record_path = getenv("GLIMPSE_RECORDING_PATH");
            data->recording = gm_recording_init(data->log, data->recording_device,
                                                record_path ? record_path :
                                                "glimpse_viewer_recording",
                                                data->overwrite_recording);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(data->playback_device ? "Unload" : "Load") &&
        !data->recording)
    {
        if (data->playback_device) {
            viewer_close_playback_device(data);

            // Wake up the recording device again
            handle_device_ready(data, data->recording_device);
        } else {
            gm_device_stop(data->recording_device);

            unref_device_frames(data);

            if (data->latest_tracking) {
                gm_tracking_unref(data->latest_tracking);
                data->latest_tracking = nullptr;
            }

            gm_context_flush(data->ctx, NULL);
            data->tracking_ready = false;

            struct gm_device_config config = {};
            config.type = GM_DEVICE_RECORDING;
            const char *record_path = getenv("GLIMPSE_RECORDING_PATH");
            config.recording.path =
                record_path ? record_path : "glimpse_viewer_recording";

            data->playback_device = gm_device_open(data->log, &config, NULL);
            gm_device_set_event_callback(data->playback_device,
                                         on_device_event_cb, data);
            data->active_device = data->playback_device;

            gm_device_commit_config(data->playback_device, NULL);
        }
    }

    ImGui::Spacing();

    ImGui::SetWindowSize(ImVec2(0, 0), ImGuiCond_Always);

    ImVec2 size = ImGui::GetWindowSize();
    ImGui::SetWindowPos(ImVec2(bounds.x + (bounds.z - size.x) / 2, 16.f),
                        ImGuiCond_FirstUseEver);

    // Make sure the window stays within bounds
    ImVec2 pos = ImGui::GetWindowPos();

    if (pos.x + size.x > bounds.x + bounds.z) {
        pos.x = (bounds.x + bounds.z) - size.x;
    } else if (pos.x < bounds.x) {
        pos.x = bounds.x;
    }
    if (pos.y + size.y > bounds.y + bounds.w) {
        pos.y = (bounds.y + bounds.w) - size.y;
    } else if (pos.y < bounds.y) {
        pos.y = bounds.y;
    }

    ImGui::SetWindowPos(pos, ImGuiCond_Always);

    ImGui::End();
}

static bool
draw_visualisation(Data *data, int x, int y, int width, int height,
                   int aspect_width, int aspect_height,
                   const char *name, GLuint tex,
                   enum gm_rotation rotation)
{
    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    ImGui::Begin(name, NULL,
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoScrollWithMouse |
                 ImGuiWindowFlags_NoCollapse |
                 ImGuiWindowFlags_NoBringToFrontOnFocus);
    bool focused = ImGui::IsWindowFocused();
    if (tex == 0) {
        return focused;
    }

    ImVec2 uv0, uv1, uv2, uv3;

    switch (rotation) {
    case GM_ROTATION_0:
        uv0 = ImVec2(0, 0);
        uv1 = ImVec2(1, 0);
        uv2 = ImVec2(1, 1);
        uv3 = ImVec2(0, 1);
        break;
    case GM_ROTATION_90:
        uv0 = ImVec2(1, 0);
        uv1 = ImVec2(1, 1);
        uv2 = ImVec2(0, 1);
        uv3 = ImVec2(0, 0);
        std::swap(aspect_width, aspect_height);
        break;
    case GM_ROTATION_180:
        uv0 = ImVec2(1, 1);
        uv1 = ImVec2(0, 1);
        uv2 = ImVec2(0, 0);
        uv3 = ImVec2(1, 0);
        break;
    case GM_ROTATION_270:
        uv0 = ImVec2(0, 1);
        uv1 = ImVec2(0, 0);
        uv2 = ImVec2(1, 0);
        uv3 = ImVec2(1, 1);
        std::swap(aspect_width, aspect_height);
        break;
    }

    ImVec2 area_size = ImGui::GetContentRegionAvail();
    adjust_aspect(area_size, aspect_width, aspect_height);

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 cur = ImGui::GetCursorScreenPos();
    draw_list->PushTextureID((void *)(intptr_t)tex);

    draw_list->PrimReserve(6, 4);
    draw_list->PrimQuadUV(ImVec2(cur.x, cur.y),
                          ImVec2(cur.x+area_size.x, cur.y),
                          ImVec2(cur.x+area_size.x, cur.y+area_size.y),
                          ImVec2(cur.x, cur.y+area_size.y),
                          uv0,
                          uv1,
                          uv2,
                          uv3,
                          ImGui::GetColorU32(ImVec4(1,1,1,1)));
    draw_list->PopTextureID();
    ImGui::End();

    return focused;
}

static void
update_tracking_buffers(Data *data)
{
    if (!data->latest_tracking) {
        return;
    }

    /*
     * Update labelled point cloud
     */
    data->n_joints = 0;
    float *joints = gm_context_predict_joint_positions(
        data->ctx, data->latest_tracking, data->last_video_frame->timestamp,
        &data->n_joints);

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

    free(joints);
}

static void
update_cloud_vis(Data *data, ImVec2 win_size, ImVec2 uiScale)
{
    const struct gm_intrinsics *video_intrinsics =
        gm_tracking_get_video_camera_intrinsics(data->latest_tracking);
    int video_width = video_intrinsics->width;
    int video_height = video_intrinsics->height;

    const struct gm_intrinsics *depth_intrinsics =
        gm_tracking_get_depth_camera_intrinsics(data->latest_tracking);
    int depth_width = depth_intrinsics->width;
    int depth_height = depth_intrinsics->height;

    // Ensure the framebuffer texture is valid
    if (!cloud_tex_valid) {
        int width = win_size.x * uiScale.x;
        int height = win_size.y * uiScale.y;

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
    glm::mat4 proj = intrinsics_to_project_matrix(depth_intrinsics, 0.01f, 10);
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
    glViewport(0, 0, win_size.x * uiScale.x, win_size.y * uiScale.y);

    glUseProgram(gl_db_program);
    glUniformMatrix4fv(gl_db_uni_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    float pt_size = ceilf((win_size.x * uiScale.x) / depth_width);
    glUniform1f(gl_db_uni_pt_size, pt_size);

    // Update camera intrinsics
    glUniform2i(gl_db_uni_depth_size,
                (GLint)depth_width,
                (GLint)depth_height);
    glUniform2f(gl_db_uni_video_size,
                (GLfloat)video_width,
                (GLfloat)video_height);
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

    glEnableVertexAttribArray(gl_db_attr_depth);
    glBindBuffer(GL_ARRAY_BUFFER, gl_db_depth_bo);
    glVertexAttribPointer(gl_db_attr_depth, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindTexture(GL_TEXTURE_2D, gl_db_vid_tex);
    glDrawArrays(GL_POINTS, 0, depth_width * depth_height);

    glDisableVertexAttribArray(gl_db_attr_depth);

    update_tracking_buffers(data);

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
        glUniform1f(gl_cloud_uni_size, pt_size * 3.f);

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
}

static bool
draw_cloud_visualisation(Data *data, ImVec2 &uiScale,
                         int x, int y, int width, int height)
{
    const struct gm_intrinsics *depth_intrinsics =
        gm_tracking_get_depth_camera_intrinsics(data->latest_tracking);
    int depth_width = depth_intrinsics->width;
    int depth_height = depth_intrinsics->height;

    bool focused = draw_visualisation(data, x, y, width, height,
                                      depth_width, depth_height,
                                      "Cloud", 0, GM_ROTATION_0);

    ImVec2 win_size = ImGui::GetContentRegionMax();
    adjust_aspect(win_size, depth_width, depth_height);
    update_cloud_vis(data, win_size, uiScale);

    ImGui::Image((void *)(intptr_t)gl_cloud_tex, win_size);

    // Handle input for cloud visualisation
    if (ImGui::IsWindowHovered()) {
        if (ImGui::IsMouseDragging()) {
            ImVec2 drag_delta = ImGui::GetMouseDragDelta();
            data->camera_rot_yx[0] += (drag_delta.x * M_PI / 180.f) * 0.2f;
            data->camera_rot_yx[1] += (drag_delta.y * M_PI / 180.f) * 0.2f;
            ImGui::ResetMouseDragDelta();
        }
    }

    ImGui::End();

    return focused;
}

static bool
draw_view(Data *data, int view, ImVec2 &uiScale,
          int x, int y, int width, int height, bool disabled)
{
    switch(view) {
    case 0:
        return draw_controls(data, x, y, width, height, disabled);
    case 1: {
        if (!data->last_video_frame) {
            return false;
        }
        const struct gm_intrinsics *video_intrinsics =
            gm_device_get_video_intrinsics(data->active_device);
        int video_width = video_intrinsics->width;
        int video_height = video_intrinsics->height;

        return draw_visualisation(data, x, y, width, height,
                                  video_width, video_height,
                                  views[view], gl_vid_tex,
                                  data->last_video_frame->camera_rotation);
    }
    case 2:
        return draw_visualisation(data, x, y, width, height,
                                  data->depth_rgb_width,
                                  data->depth_rgb_height,
                                  views[view], gl_depth_rgb_tex,
                                  GM_ROTATION_0);
    case 3:
        return draw_visualisation(data, x, y, width, height,
                                  data->normals_rgb_width,
                                  data->normals_rgb_height,
                                  views[view], gl_normals_rgb_tex,
                                  GM_ROTATION_0);
    case 4:
        return draw_visualisation(data, x, y, width, height,
                                  data->nclusters_rgb_width,
                                  data->nclusters_rgb_height,
                                  views[view], gl_nclusters_rgb_tex,
                                  GM_ROTATION_0);
    case 5:
        return draw_visualisation(data, x, y, width, height,
                                  data->cclusters_rgb_width,
                                  data->cclusters_rgb_height,
                                  views[view], gl_cclusters_rgb_tex,
                                  GM_ROTATION_0);
    case 6:
        return draw_visualisation(data, x, y, width, height,
                                  data->labels_rgb_width,
                                  data->labels_rgb_height,
                                  views[view], gl_labels_tex,
                                  GM_ROTATION_0);
    case 7:
        if (!data->latest_tracking) {
            return false;
        }
        return draw_cloud_visualisation(data, uiScale,
                                        x, y, width, height);
    }

    return false;
}

static void
draw_ui(Data *data)
{
    static int cloud_view = ARRAY_LEN(views) - 1;
    static int main_view = 1;
    int current_view = main_view;

    ProfileScopedSection(DrawIMGUI, ImGuiControl::Profiler::Dark);

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 uiScale = io.DisplayFramebufferScale;
    ImVec2 origin = io.DisplayVisibleMin;
    ImVec2 win_size = ImVec2(io.DisplayVisibleMax.x - io.DisplayVisibleMin.x,
                             io.DisplayVisibleMax.y - io.DisplayVisibleMin.y);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

    bool skip_controls = false;
    if (current_view != 0) {
        // Draw playback controls if UI controls isn't the main view
        draw_playback_controls(data, ImVec4(0, 0, win_size.x, win_size.y));
    }
    if (win_size.x >= 1024 && win_size.y >= 600) {
        // Draw control panel on the left if we have a large window
        draw_controls(data, origin.x, origin.y,
                      TOOLBAR_WIDTH + origin.x, win_size.y - origin.y, false);

        win_size.x -= TOOLBAR_WIDTH;
        origin.x += TOOLBAR_WIDTH;

        skip_controls = true;
    }

    // Draw sub-views on the axis with the most space
    int view = skip_controls ? 1 : 0;
    int n_views = ARRAY_LEN(views) - (skip_controls ? 1 : 0);
    for (int s = 0; s <= (n_views - 1) / MAX_VIEWS; ++s) {
        int subview_width, subview_height;
        if (win_size.x > win_size.y) {
            subview_height = win_size.y / MAX_VIEWS;
            subview_width = data->depth_rgb_height ?
                subview_height * (data->video_rgb_width /
                                  (float)data->video_rgb_height) :
                subview_height;
        } else {
            subview_width = win_size.x / MAX_VIEWS;
            subview_height = data->depth_rgb_width ?
                subview_width * (data->video_rgb_height /
                                 (float)data->video_rgb_width) :
                subview_width;
        }
        for (int i = 0; i < MAX_VIEWS; ++i, ++view) {
            if (view == current_view) {
                ++view;
            }
            if (view >= (int)ARRAY_LEN(views)) {
                break;
            }

            int x, y;
            if (win_size.x > win_size.y) {
                x = origin.x + win_size.x - subview_width;
                y = origin.y + (subview_height * i);
            } else {
                y = origin.y + (win_size.y - subview_height);
                x = origin.x + (subview_width * i);
            }

            if (draw_view(data, view, uiScale, x, y,
                          subview_width, subview_height, view == 0)) {
                main_view = view;
            }
        }

        if (win_size.x > win_size.y) {
            win_size.x -= subview_width;
        } else {
            win_size.y -= subview_height;
        }
    }

    // Draw the main view in the remaining space in the center
    draw_view(data, current_view, uiScale, origin.x, origin.y,
              win_size.x, win_size.y, false);

    ImGui::PopStyleVar();

    // Draw profiler window always-on-top
    ImGui::SetNextWindowPos(origin, ImGuiCond_Once);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ProfileDrawUI();

    ImGui::Render();

    // If we've toggled between the cloud view, invalidate the texture so
    // it gets recreated at the right size next time it's displayed.
    if (main_view != current_view &&
        (main_view == cloud_view || current_view == cloud_view)) {
        cloud_tex_valid = false;
    }
}

/* If we've already requested gm_device for a frame then this won't submit
 * a request that downgrades the requirements
 */
static void
request_device_frame(Data *data, uint64_t requirements)
{
    uint64_t new_requirements = data->pending_frame_requirements | requirements;

    if (data->pending_frame_requirements != new_requirements) {
        gm_device_request_frame(data->active_device, new_requirements);
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

    {
        ProfileScopedSection(GetLatestFrame);
        /* NB: gm_device_get_latest_frame will give us a _ref() */
        gm_frame *device_frame = gm_device_get_latest_frame(data->active_device);
        if (!device_frame) {
            return;
        }
        upload = true;

        if (device_frame->depth) {
            if (data->last_depth_frame) {
                gm_frame_unref(data->last_depth_frame);
            }
            gm_frame_ref(device_frame);
            data->last_depth_frame = device_frame;
            data->pending_frame_requirements &= ~GM_REQUEST_FRAME_DEPTH;
        }

        if (device_frame->video) {
            if (data->last_video_frame) {
                gm_frame_unref(data->last_video_frame);
            }
            gm_frame_ref(device_frame);
            data->last_video_frame = device_frame;
            data->pending_frame_requirements &= ~GM_REQUEST_FRAME_VIDEO;
        }

        if (data->recording) {
            gm_recording_save_frame(data->recording, device_frame);
        }

        gm_frame_unref(device_frame);
    }

    if (data->context_needs_frame &&
        data->last_depth_frame && data->last_video_frame) {
        ProfileScopedSection(FwdContextFrame);

        // Combine the two video/depth frames into a single frame for gm_context
        if (data->last_depth_frame != data->last_video_frame) {
            struct gm_frame *full_frame =
                gm_device_combine_frames(data->active_device,
                                         std::max(
                                             data->last_video_frame->timestamp,
                                             data->last_depth_frame->timestamp),
                                         data->last_depth_frame,
                                         data->last_video_frame);

            // We don't need the individual frames any more
            gm_frame_unref(data->last_depth_frame);
            gm_frame_unref(data->last_video_frame);

            data->last_depth_frame = full_frame;
            data->last_video_frame = gm_frame_ref(full_frame);
        }

        data->context_needs_frame =
            !gm_context_notify_frame(data->ctx, data->last_depth_frame);

        // We don't want to send duplicate frames to tracking, so discard now
        gm_frame_unref(data->last_depth_frame);
        data->last_depth_frame = NULL;
    }

    data->device_frame_ready = false;

    {
        ProfileScopedSection(DeviceFrameRequest);

        /* immediately request a new frame since we want to render the camera
         * at the native capture rate, even though we might not be tracking
         * at that rate.
         *
         * Similarly, if we're recording, request depth frames so that we can
         * record at a rate that exceeds the tracking rate.
         *
         * Note: the requirements may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, data->recording ?
                             (GM_REQUEST_FRAME_DEPTH | GM_REQUEST_FRAME_VIDEO) :
                             GM_REQUEST_FRAME_VIDEO);
    }

    if (upload && data->last_video_frame) {
        const struct gm_intrinsics *video_intrinsics =
            gm_device_get_video_intrinsics(data->active_device);
        int video_width = video_intrinsics->width;
        int video_height = video_intrinsics->height;

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

        void *video_front = data->last_video_frame->video->data;
        enum gm_format video_format = data->last_video_frame->video_format;

        switch (video_format) {
        case GM_FORMAT_LUMINANCE_U8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                         video_width, video_height,
                         0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_RGB_U8:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                         video_width, video_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, video_front);
            break;

        case GM_FORMAT_RGBX_U8:
        case GM_FORMAT_RGBA_U8:
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

    uint8_t *depth_rgb = NULL;
    gm_tracking_create_rgb_depth(data->latest_tracking,
                                 &data->depth_rgb_width,
                                 &data->depth_rgb_height,
                                 &depth_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->depth_rgb_width, data->depth_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, depth_rgb);
    free(depth_rgb);

    /* Update depth buffer and colour buffer */
    const float *depth = gm_tracking_get_depth(data->latest_tracking);
    glBindBuffer(GL_ARRAY_BUFFER, gl_db_depth_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float) * data->depth_rgb_width * data->depth_rgb_height,
                 depth, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, gl_db_vid_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    uint8_t *video_rgb = NULL;
    gm_tracking_create_rgb_video(data->latest_tracking,
                                 &data->video_rgb_width,
                                 &data->video_rgb_height,
                                 &video_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->video_rgb_width, data->video_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, video_rgb);
    free(video_rgb);

    /* Update normals buffer */
    glBindTexture(GL_TEXTURE_2D, gl_normals_rgb_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    uint8_t *normals_rgb = NULL;
    gm_tracking_create_rgb_normals(data->latest_tracking,
                                   &data->normals_rgb_width,
                                   &data->normals_rgb_height,
                                   &normals_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->normals_rgb_width, data->normals_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, normals_rgb);
    free(normals_rgb);

    /* Update normal clusters buffer */
    glBindTexture(GL_TEXTURE_2D, gl_nclusters_rgb_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    uint8_t *nclusters_rgb = NULL;
    gm_tracking_create_rgb_normal_clusters(data->latest_tracking,
                                           &data->nclusters_rgb_width,
                                           &data->nclusters_rgb_height,
                                           &nclusters_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->nclusters_rgb_width, data->nclusters_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, nclusters_rgb);
    free(nclusters_rgb);

    /* Update candidate clusters buffer */
    glBindTexture(GL_TEXTURE_2D, gl_cclusters_rgb_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    uint8_t *cclusters_rgb = NULL;
    gm_tracking_create_rgb_candidate_clusters(data->latest_tracking,
                                              &data->cclusters_rgb_width,
                                              &data->cclusters_rgb_height,
                                              &cclusters_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->cclusters_rgb_width, data->cclusters_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, cclusters_rgb);
    free(cclusters_rgb);

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

    uint8_t *labels_rgb = NULL;
    gm_tracking_create_rgb_label_map(data->latest_tracking,
                                     &data->labels_rgb_width,
                                     &data->labels_rgb_height,
                                     &labels_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->labels_rgb_width, data->labels_rgb_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, labels_rgb);
    free(labels_rgb);
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

    // When flushing the context, we can end up with notified tracking but
    // no tracking to pick up
    if (!data->latest_tracking) {
        return;
    }

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
handle_device_ready(Data *data, struct gm_device *dev)
{
    gm_debug(data->log, "%s device ready\n",
            dev == data->playback_device ? "Playback" : "Default");

    if (!data->gl_initialized) {
        init_viewer_opengl(data);
    }

    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(dev);
    gm_context_set_depth_camera_intrinsics(data->ctx, depth_intrinsics);

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(dev);
    gm_context_set_video_camera_intrinsics(data->ctx, video_intrinsics);

    /*gm_context_set_depth_to_video_camera_extrinsics(data->ctx,
      gm_device_get_depth_to_video_extrinsics(dev));*/

    uint64_t old_reqs = data->pending_frame_requirements;
    data->pending_frame_requirements = 0;
    gm_device_start(dev);
    gm_context_enable(data->ctx);
    if (old_reqs) {
        request_device_frame(data, old_reqs);
    }
}

static void
handle_device_event(Data *data, struct gm_device_event *event)
{
    // Ignore unexpected device events
    if (event->device != data->active_device) {
        gm_device_event_free(event);
        return;
    }

    switch (event->type) {
    case GM_DEV_EVENT_READY:
        handle_device_ready(data, event->device);
        break;
    case GM_DEV_EVENT_FRAME_READY:
        if (event->frame_ready.met_requirements &
            data->pending_frame_requirements) {
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
        gm_debug(data->log, "Requesting frame\n");
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
event_loop_iteration(Data *data)
{
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

}

#ifdef USE_GLFM
static void
surface_created_cb(GLFMDisplay *display, int width, int height)
{
    Data *data = (Data *)glfmGetUserData(display);

    gm_debug(data->log, "Surface created (%dx%d)", width, height);

    if (!data->surface_created) {
        init_basic_opengl(data);
        data->surface_created = true;
    }

    data->win_width = width;
    data->win_height = height;
    cloud_tex_valid = false;
}

static void
surface_destroyed_cb(GLFMDisplay *display)
{
    Data *data = (Data *)glfmGetUserData(display);
    gm_debug(data->log, "Surface destroyed");
    data->surface_created = false;
    cloud_tex_valid = false;
}

static void
app_focus_cb(GLFMDisplay *display, bool focused)
{
    Data *data = (Data *)glfmGetUserData(display);
    gm_debug(data->log, focused ? "Focused" : "Unfocused");

    if (focused) {
        if (data->playback_device) {
            gm_device_start(data->playback_device);
        } else {
            gm_device_start(data->recording_device);
        }
    } else {
        if (data->playback_device) {
            gm_device_stop(data->playback_device);
        } else {
            gm_device_stop(data->recording_device);
        }
    }
}

static void
frame_cb(GLFMDisplay* display, double frameTime)
{
    Data *data = (Data*)glfmGetUserData(display);

    if (permissions_check_passed) {
        if (!data->initialized)
            viewer_init(data);

        ProfileNewFrame();
        ProfileScopedSection(Frame);
        event_loop_iteration(data);

        {
            ProfileScopedSection(Redraw);

            glViewport(0, 0, data->win_width, data->win_height);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplGlfmGLES3_NewFrame(display, frameTime);
            draw_ui(data);
        }

    } else if (permissions_check_failed) {
        /* At least some visual feedback that we failed to
         * acquire the permissions we need...
         */
        glClearColor(1.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        glClear(GL_COLOR_BUFFER_BIT);
    }
}
#endif

#ifdef USE_GLFW
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

        event_loop_iteration(data);

        {
            ProfileScopedSection(Redraw);

            glViewport(0, 0, data->win_width, data->win_height);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplGlfwGLES3_NewFrame();
            draw_ui(data);
        }

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
#endif

static void
on_khr_debug_message_cb(GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum gl_severity,
                        GLsizei length,
                        const GLchar *message,
                        void *user_data)
{
    Data *data = (Data *)user_data;

    switch (gl_severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        gm_log(data->log, GM_LOG_ERROR, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        gm_log(data->log, GM_LOG_WARN, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        gm_log(data->log, GM_LOG_WARN, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        gm_log(data->log, GM_LOG_INFO, "Viewer GL", "%s", message);
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
on_device_event_cb(struct gm_device_event *device_event,
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

/* Initialize enough OpenGL state to handle rendering before being
 * notified that the Glimpse device is 'ready' (i.e. before it's
 * possible to query camera intrinsics)
 */
static void
init_basic_opengl(Data *data)
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClearStencil(0);

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
}

static void
init_viewer_opengl(Data *data)
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
        "  float ty = ((dy * fy / depth) + (video_size.y - cy)) / video_size.y;\n"

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

        /* XXX: Mesa bug? glsl es 300 should support texture() but this isn't
         * working with Mesa (and it's not complaining about the
         * "#version 300 es")
         */
#ifdef __ANDROID__
        "  color = texture(texture, v_tex_coord.st);\n"
#else
        "  color = texture2D(texture, v_tex_coord.st);\n"
#endif
        "}\n";


    // Create depth-buffer point shader
    gl_db_program = gm_gl_create_program(data->log,
                                         vertShaderDepth,
                                         fragShaderDepth,
                                         NULL);

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

    glUseProgram(0);

    // Create point-cloud shader
    gl_cloud_program = gm_gl_create_program(data->log,
                                            vertShaderCloud,
                                            fragShaderCloud,
                                            NULL);

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

    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_vid_tex);
    glBindTexture(GL_TEXTURE_2D, gl_vid_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_normals_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_normals_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_nclusters_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_nclusters_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_cclusters_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_cclusters_rgb_tex);
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

    data->gl_initialized = true;
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
    Data *data = (Data *)user_data;
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
    Data *data = (Data *)user_data;

    if (data->log_fp) {
        fprintf(data->log_fp, "ABORT\n");
        fflush(data->log_fp);
        fclose(data->log_fp);
    }

    abort();
}

#ifdef USE_GLFM
static void
init_winsys_glfm(Data *data, GLFMDisplay *display)
{
    glfmSetDisplayConfig(display,
                         GLFMRenderingAPIOpenGLES3,
                         GLFMColorFormatRGBA8888,
                         GLFMDepthFormatNone,
                         GLFMStencilFormatNone,
                         GLFMMultisampleNone);
    glfmSetDisplayChrome(display,
                         GLFMUserInterfaceChromeNavigationAndStatusBar);
    glfmSetUserData(display, data);
    glfmSetSurfaceCreatedFunc(display, surface_created_cb);
    glfmSetSurfaceResizedFunc(display, surface_created_cb);
    glfmSetSurfaceDestroyedFunc(display, surface_destroyed_cb);
    glfmSetAppFocusFunc(display, app_focus_cb);
    glfmSetMainLoopFunc(display, frame_cb);

    ImGui_ImplGlfmGLES3_Init(display, true);
}
#endif

#ifdef USE_GLFW
static void
init_winsys_glfw(Data *data)
{
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW, OpenGL windows system library\n");
        exit(1);
    }

    data->win_width = 1280;
    data->win_height = 720;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  0) ;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);

    data->window = glfwCreateWindow(data->win_width,
                                    data->win_height,
                                    "Glimpse Viewer", NULL, NULL);
    if (!data->window) {
        fprintf(stderr, "Failed to create window\n");
        exit(1);
    }

    glfwSetWindowUserPointer(data->window, data);

    glfwSetFramebufferSizeCallback(data->window, on_window_fb_size_change_cb);

    glfwMakeContextCurrent(data->window);
    glfwSwapInterval(1);

    glfwSetErrorCallback(on_glfw_error_cb);

    ImGui_ImplGlfwGLES3_Init(data->window, false /* don't install callbacks */);

    /* will chain on to ImGui_ImplGlfwGLES3_KeyCallback... */
    glfwSetKeyCallback(data->window, on_key_input_cb);
    glfwSetMouseButtonCallback(data->window,
                               ImGui_ImplGlfwGLES3_MouseButtonCallback);
    glfwSetScrollCallback(data->window, ImGui_ImplGlfwGLES3_ScrollCallback);
    glfwSetCharCallback(data->window, ImGui_ImplGlfwGLES3_CharCallback);

    init_basic_opengl(data);
}
#endif // USE_GLFW

static void __attribute__((unused))
viewer_destroy(Data *data)
{
    if (data->playback_device) {
        viewer_close_playback_device(data);
    }

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
    gm_device_stop(data->recording_device);

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

    unref_device_frames(data);

    gm_device_close(data->recording_device);

    json_value_free(data->joint_map);

    gm_logger_destroy(data->log);

    delete data->events_front;
    delete data->events_back;
    delete data;

    ProfileShutdown();
}

static void
viewer_init(Data *data)
{
    ImGuiIO& io = ImGui::GetIO();

    char *open_err = NULL;
    struct gm_asset *font_asset = gm_asset_open(data->log,
                                                "Roboto-Medium.ttf",
                                                GM_ASSET_MODE_BUFFER,
                                                &open_err);
    if (font_asset) {
        const void *buf = gm_asset_get_buffer(font_asset);

        unsigned len = gm_asset_get_length(font_asset);
        void *buf_copy = ImGui::MemAlloc(len);
        memcpy(buf_copy, buf, len);

        ImVec2 uiScale = io.DisplayFramebufferScale;
        io.Fonts->AddFontFromMemoryTTF(buf_copy, 16.f, 16.f * uiScale.x);
        gm_asset_close(font_asset);
    } else {
        gm_error(data->log, "%s", open_err);
        exit(1);
    }

    const char *n_frames_env = getenv("GLIMPSE_RECORD_N_JOINT_FRAMES");
    if (n_frames_env) {
        data->joints_recording_val = json_value_init_array();
        data->joints_recording = json_array(data->joints_recording_val);
        data->requested_recording_len = strtoull(n_frames_env, NULL, 10);
    }

    // TODO: Might be nice to be able to retrieve this information via the API
    //       rather than reading it separately here.
    struct gm_asset *joint_map_asset = gm_asset_open(data->log,
                                                     "joint-map.json",
                                                     GM_ASSET_MODE_BUFFER,
                                                     &open_err);
    if (joint_map_asset) {
        const void *buf = gm_asset_get_buffer(joint_map_asset);
        data->joint_map = json_parse_string((const char *)buf);
        gm_asset_close(joint_map_asset);
    } else {
        gm_error(data->log, "%s", open_err);
        exit(1);
    }

    // Count the number of bones defined by connections in the joint map.
    data->n_bones = 0;
    for (size_t i = 0; i < json_array_get_count(json_array(data->joint_map));
         i++) {
        JSON_Object *joint =
            json_array_get_object(json_array(data->joint_map), i);
        data->n_bones += json_array_get_count(
            json_object_get_array(joint, "connections"));
    }

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    data->ctx = gm_context_new(data->log, NULL);

    gm_context_set_event_callback(data->ctx, on_event_cb, data);

    struct gm_asset *config_asset =
        gm_asset_open(data->log,
                      "glimpse-config.json", GM_ASSET_MODE_BUFFER, &open_err);
    if (config_asset) {
        const char *buf = (const char *)gm_asset_get_buffer(config_asset);
        gm_config_load(data->log, buf, gm_context_get_ui_properties(data->ctx));
        gm_asset_close(config_asset);
    } else {
        gm_warn(data->log, "Failed to open glimpse-config.json: %s", open_err);
        free(open_err);
    }

    struct gm_device_config config = {};
#ifdef USE_TANGO
    config.type = GM_DEVICE_TANGO;
#else
    config.type = GM_DEVICE_KINECT;
#endif
    data->recording_device = gm_device_open(data->log, &config, NULL);
    data->active_device = data->recording_device;
    gm_device_set_event_callback(data->recording_device, on_device_event_cb, data);
#ifdef __ANDROID__
    gm_device_attach_jvm(data->recording_device, android_jvm_singleton);
#endif
    gm_device_commit_config(data->recording_device, NULL);

    data->initialized = true;
}


#ifdef USE_GLFM
void
glfmMain(GLFMDisplay *display)
#else
int
main(int argc, char **argv)
#endif
{
    Data *data = new Data();

#ifdef __ANDROID__
#define ANDROID_ASSETS_ROOT "/sdcard/GlimpseUnity"
    setenv("GLIMPSE_ASSETS_ROOT", ANDROID_ASSETS_ROOT, true);
    setenv("GLIMPSE_RECORDING_PATH", ANDROID_ASSETS_ROOT "/ViewerRecording",
           true);
    setenv("FAKENECT_PATH", ANDROID_ASSETS_ROOT "/FakeRecording", true);
    data->log_fp = fopen(ANDROID_ASSETS_ROOT "/glimpse.log", "w");
#else
    data->log_fp = stderr;
#endif

    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

#ifdef USE_GLFW
    init_winsys_glfw(data);
#endif
#ifdef USE_GLFM
    init_winsys_glfm(data, display);
#endif

    data->events_front = new std::vector<struct event>();
    data->events_back = new std::vector<struct event>();
    data->focal_point = glm::vec3(0.0, 0.0, 2.5);
    data->overwrite_recording = true;

#ifdef __ANDROID__
    // Quick hack to make scrollbars a bit more usable on small devices
    ImGui::GetStyle().ScrollbarSize *= 2;
#endif

#ifdef USE_GLFW
    viewer_init(data);

    event_loop(data);

    viewer_destroy(data);

    ImGui_ImplGlfwGLES3_Shutdown();
    glfwDestroyWindow(data->window);
    glfwTerminate();

    return 0;
#endif
}

#ifdef __ANDROID__
extern "C" jint
JNI_OnLoad(JavaVM *vm, void *reserved)
{
    android_jvm_singleton = vm;

    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL
Java_com_impossible_glimpse_GlimpseNativeActivity_OnPermissionsCheckResult(
    JNIEnv *env, jclass type, jboolean permission)
{
    /* Just wait for the next frame to check these */
    if (permission) {
        permissions_check_passed = true;
    } else
        permissions_check_failed = true;
}
#endif
