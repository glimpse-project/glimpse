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
#include <ctype.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>

#include <math.h>

#include <pthread.h>

#include <list>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>

#include <epoxy/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_internal.h> // For PushItemFlags(ImGuiItemFlags_Disabled)

#ifdef __ANDROID__
#    include <android/log.h>
#    include <jni.h>
#endif

#if defined(__APPLE__)
#    include <TargetConditionals.h>
#else
#    define TARGET_OS_MAC 0
#    define TARGET_OS_IOS 0
#    define TARGET_OS_OSX 0
#endif

#if TARGET_OS_IOS == 1
#    include "ios_utils.h"
#endif

#ifdef USE_GLFM
#    define GLFM_INCLUDE_NONE
#    include <glfm.h>
#    include <imgui_impl_glfm_gles3.h>
#else
#    define GLFW_INCLUDE_NONE
#    include <GLFW/glfw3.h>
#    include <imgui_impl_glfw_gles3.h>
#    include <getopt.h>
#endif

#if TARGET_OS_OSX == 1
#define GLSL_SHADER_VERSION "#version 400\n"
#else
#define GLSL_SHADER_VERSION "#version 300 es\n"
#endif

#ifdef USE_TANGO
#include <tango_client_api.h>
#include <tango_support_api.h>
#endif

#include <profiler.h>

#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_device.h"
#include "glimpse_record.h"
#include "glimpse_assets.h"
#include "glimpse_gl.h"
#include "glimpse_target.h"

#undef GM_LOG_CONTEXT
#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Viewer"
#else
#define GM_LOG_CONTEXT "viewer"
#endif

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))
#define LOOP_INDEX(x,y) ((x)[(y) % ARRAY_LEN(x)])

#define TOOLBAR_WIDTH 400
#define MAX_VIEWS 5

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)


typedef struct _Data Data;

typedef void (*control_prop_draw_callback_t)(Data *data,
                                             struct gm_ui_properties *props,
                                             struct gm_ui_property *prop,
                                             const char *readable_name);

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

typedef struct {
    float x;
    float y;
    float z;
    uint32_t rgba;
} XYZRGBA;

struct debug_image {
    GLuint gl_tex;
    int width;
    int height;
};

#define MAX_IMAGES_PER_STAGE 5
struct stage_textures {
    struct debug_image images[MAX_IMAGES_PER_STAGE];
};

typedef struct {
    GLuint joints_bo;
    GLuint bones_bo;
    int n_joints;
    int n_bones;
} GLSkeleton;

struct _Data
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

    /* Some GL state is re-initialized each time we switch devices */
    bool device_gl_initialized;

    struct gm_context *ctx;

#ifdef USE_GLFW
    GLFWwindow *window;
#else
    bool surface_created;
#endif
    int win_width;
    int win_height;

    /* Normally this is 'false' and we show lots of intermediate debug buffers
     * but e.g. if running on Android with Tango then we try to more closely
     * represent a 'real' augmented reality app with fullscreen video plus a
     * skeleton overlay so we can better judge the base-line performance we
     * can expect to achieve for these kinds of applications.
     * (uploading all of the debug textures can significantly impact the
     * runtime performance, e.g. taking > 100ms each time we get a tracking
     * update)
     */
    bool realtime_ar_mode;

    bool show_skeleton;
    bool show_view_cam_controls;
    bool show_profiler;
    bool show_joint_summary;

    int stage_stats_mode;

    /* In realtime mode, we use predicted joint positions so that the
     * presented skeleton keeps up with the video. This allows us to add a
     * synthetic delay to the timestamp we request in this mode, which adds
     * some lag, but improves the quality of the positions as it doesn't need
     * to extrapolate so far into the future.
     */
    int prediction_delay;

    float view_zoom;
    glm::vec3 focal_point;
    float camera_rot_yx[2];

    JSON_Value *joint_map;

    /* When we request gm_device for a frame we set a buffers_mask for what the
     * frame should include. We track the buffers_mask so we avoid sending
     * subsequent frame requests that would downgrade the buffers_mask
     */
    uint64_t pending_frame_buffers_mask;

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
    bool track_while_recording;
    struct gm_recording *recording;
    struct gm_device *recording_device;
    std::vector<char *> recordings;
    std::vector<char *> recording_names;
    int selected_playback_recording;

    struct gm_device *playback_device;

    struct gm_device *active_device;

    /* Events from the gm_context and gm_device apis may be delivered via any
     * arbitrary thread which we don't want to block, and at a time where
     * the gm_ apis may not be reentrant due to locks held during event
     * notification
     */
    pthread_mutex_t event_queue_lock;
    pthread_cond_t event_notify_cond;
    std::vector<struct event> *events_back;
    std::vector<struct event> *events_front;

    JSON_Value *joints_recording_val;
    JSON_Array *joints_recording;
    int requested_recording_len;

    GLuint video_program;
    GLuint video_quad_attrib_bo;

    /* Even though glEnable/DisableVertexAttribArray take unsigned integers,
     * these are signed because GL's glGetAttribLocation api returns attribute
     * locations as signed values where -1 means the attribute isn't
     * active. ...!?
     */
    GLint video_quad_attrib_pos;
    GLint video_quad_attrib_tex_coords;


    GLuint cloud_fbo;
    GLuint cloud_depth_renderbuf;
    GLuint cloud_fbo_tex;
    bool cloud_fbo_valid;

    GLuint cloud_program;
    GLuint cloud_uniform_mvp;
    GLuint cloud_uniform_pt_size;

    GLuint cloud_bo;
    GLint cloud_attr_pos;
    GLint cloud_attr_col;
    int n_cloud_points;
    struct gm_intrinsics cloud_intrinsics;

    GLuint lines_bo;
    GLint lines_attr_pos;
    GLint lines_attr_col;
    int n_lines;

    GLSkeleton skel_gl;

    struct gm_target *target;
    float target_error;
    bool target_progress;
    bool target_resize;
    GLSkeleton target_skel_gl;

    int selected_target;
    std::vector<char *> targets;
    std::vector<char *> target_names;

    GLuint ar_video_tex_sampler;
    std::vector<GLuint> ar_video_queue;
    int ar_video_queue_len;
    int ar_video_queue_pos;

    std::vector<struct stage_textures> stage_textures;
    int current_stage;

    /* Overrides drawing specific property widgets... */

    /* Hmm, It doesn't seem to Just Work (tm) to declare a type safe function
     * pointer value so we just use void*... strange; I distinctly remember the
     * C++ fanatic telling me something about improved type safety.. blah
     * blah...
     */
    std::unordered_map<std::string, void*> dev_control_overrides;
    std::unordered_map<std::string, void*> ctx_control_overrides;
    std::unordered_map<std::string, void*> stage_control_overrides;
};

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

char *glimpse_recordings_path;
char *glimpse_targets_path;

static bool pause_profile;

#ifdef USE_GLFM
static bool permissions_check_failed;
static bool permissions_check_passed;
#endif

static const char *log_filename_opt = NULL;

#ifdef USE_FREENECT
static enum gm_device_type device_type_opt = GM_DEVICE_KINECT;
#else
static enum gm_device_type device_type_opt = GM_DEVICE_NULL;
#endif

static char *device_recording_opt;

static void viewer_init(Data *data);

static void init_basic_opengl(Data *data);
static void init_viewer_opengl(Data *data);
static void init_device_opengl(Data *data);
static void deinit_device_opengl(Data *data);

static void handle_device_ready(Data *data, struct gm_device *dev);
static void on_device_event_cb(struct gm_device_event *device_event,
                               void *user_data);


/* Copied from glimpse_rdt.cc
 *
 * The longest format is like "00:00:00" which needs up to 9 bytes but notably
 * gcc complains if buf < 14 bytes, so rounding up to power of two for neatness.
 */
char *
format_duration_s16(uint64_t duration_ns, char buf[16])
{
    if (duration_ns > 1000000000) {
        const uint64_t hour_ns = 1000000000ULL*60*60;
        const uint64_t min_ns = 1000000000ULL*60;
        const uint64_t sec_ns = 1000000000ULL;

        uint64_t hours = duration_ns / hour_ns;
        duration_ns -= hours * hour_ns;
        uint64_t minutes = duration_ns / min_ns;
        duration_ns -= minutes * min_ns;
        uint64_t seconds = duration_ns / sec_ns;
        snprintf(buf, 16, "%02d:%02d:%02d", (int)hours, (int)minutes, (int)seconds);
    } else if (duration_ns > 1000000) {
        uint64_t ms = duration_ns / 1000000;
        snprintf(buf, 16, "%dms", (int)ms);
    } else if (duration_ns > 1000) {
        uint64_t us = duration_ns / 1000;
        snprintf(buf, 16, "%dus", (int)us);
    } else {
        snprintf(buf, 16, "%dns", (int)duration_ns);
    }

    return buf;
}

static void
make_readable_name(const char *symbolic_name,
                   char *readable_dst,
                   int readable_dst_len)
{
    int n_end = readable_dst_len - 1;
    bool seen_space = true;
    int n = 0;
    for (n = 0; n < n_end && symbolic_name[n]; n++) {
        int c = symbolic_name[n];
        if (c == '_' || c == '-' || c == ' ') {
            seen_space = true;
            readable_dst[n] = ' ';
        } else if (seen_space) {
            readable_dst[n] = toupper(c);
            seen_space = false;
        } else
            readable_dst[n] = c;
    }
    readable_dst[n] = '\0';
}

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

static glm::mat4
intrinsics_to_zoomed_project_matrix(const struct gm_intrinsics *intrinsics,
                                    float near, float far,
                                    float zoom)
{
  float width = intrinsics->width;
  float height = intrinsics->height;

  float scalex = near / intrinsics->fx;
  float scaley = near / intrinsics->fy;

  float offsetx = (intrinsics->cx - width / 2.0) * scalex;
  float offsety = (intrinsics->cy - height / 2.0) * scaley;

  float inverse_zoom = 1.0f / zoom;

  return glm::frustum(inverse_zoom * (scalex * -width / 2.0f - offsetx), // left
                      inverse_zoom * (scalex *  width / 2.0f - offsetx), // right
                      inverse_zoom * (scaley * -height / 2.0f - offsety), // bottom
                      inverse_zoom * (scaley *  height / 2.0f - offsety), // top
                      near, far);
}

static bool
index_files(Data *data,
            const char *match,
            const char *root, const char *subdir,
            std::vector<char *> &files,
            std::vector<char *> &names, char **err,
            int recurse = 1)
{
    struct dirent *entry;
    struct stat st;
    DIR *dir;
    bool ret = true;

    char full_path[512];
    xsnprintf(full_path, sizeof(full_path), "%s/%s", root, subdir);
    if (!(dir = opendir(full_path))) {
        gm_throw(data->log, err, "Failed to open directory %s\n", full_path);
        return false;
    }

    while ((entry = readdir(dir)) != NULL) {
        char cur_full_path[512];
        char cur_rel_path[512];

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;

        xsnprintf(cur_full_path, sizeof(cur_full_path), "%s/%s/%s",
                  root, subdir, entry->d_name);
        xsnprintf(cur_rel_path, sizeof(cur_rel_path), "%s/%s",
                  subdir, entry->d_name);

        stat(cur_full_path, &st);
        if (S_ISDIR(st.st_mode)) {
            if (recurse > 0 &&
                !index_files(data, match, root,
                             cur_rel_path, files, names, err, recurse - 1))
            {
                ret = false;
                break;
            }
        } else if (strlen(subdir) &&
                   strcmp(entry->d_name, match) == 0) {
            files.push_back(strdup(cur_rel_path));
            char *cur_dir = basename(dirname(cur_full_path));
            names.push_back(strdup(cur_dir));
        }
    }

    closedir(dir);

    return ret;
}

static void
index_recordings(Data *data)
{
    data->recordings.clear();
    data->recording_names.clear();

    char *index_err = NULL;
    index_files(data,
                "glimpse_recording.json",
                glimpse_recordings_path,
                "", // subdirectory
                data->recordings,
                data->recording_names,
                &index_err);
    if (index_err) {
        gm_error(data->log, "Failed to index recordings: %s", index_err);
        free(index_err);
    }
}

static void
index_targets(Data *data)
{
    data->targets.clear();
    data->target_names.clear();

    char *index_err = NULL;
    index_files(data,
                "glimpse_target.index",
                glimpse_targets_path,
                "",
                data->targets,
                data->target_names,
                &index_err);
    if (index_err) {
        gm_error(data->log, "Failed to index targets: %s", index_err);
        free(index_err);
    }
}

static void
draw_int_property(Data *data,
                  struct gm_ui_properties *props,
                  struct gm_ui_property *prop,
                  const char *readable_name)
{
    int current_val = gm_prop_get_int(prop), save_val = current_val;
    ImGui::SliderInt(readable_name, &current_val,
                     prop->int_state.min, prop->int_state.max);
    if (current_val != save_val)
        gm_prop_set_int(prop, current_val);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", prop->desc);
}

static void
draw_enum_property(Data *data,
                   struct gm_ui_properties *props,
                   struct gm_ui_property *prop,
                   const char *readable_name)
{
    int current_enumerant = 0, save_enumerant = 0;
    int current_val = gm_prop_get_enum(prop);

    for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
        if (prop->enum_state.enumerants[i].val == current_val) {
            current_enumerant = save_enumerant = i;
            break;
        }
    }

    std::vector<const char*> labels(prop->enum_state.n_enumerants);
    for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
        labels[i] = prop->enum_state.enumerants[i].name;
    }

    ImGui::Combo(readable_name, &current_enumerant, labels.data(),
                 labels.size());

    if (current_enumerant != save_enumerant) {
        int e = current_enumerant;
        gm_prop_set_enum(prop, prop->enum_state.enumerants[e].val);
    }

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", prop->desc);
}

static void
draw_bool_property(Data *data,
                   struct gm_ui_properties *props,
                   struct gm_ui_property *prop,
                   const char *readable_name)
{
    bool current_val = gm_prop_get_bool(prop),
         save_val = current_val;
    ImGui::Checkbox(readable_name, &current_val);
    if (current_val != save_val)
        gm_prop_set_bool(prop, current_val);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", prop->desc);
}

static void
draw_float_property(Data *data,
                    struct gm_ui_properties *props,
                    struct gm_ui_property *prop,
                    const char *readable_name)
{
    float current_val = gm_prop_get_float(prop), save_val = current_val;
    ImGui::SliderFloat(readable_name, &current_val,
                       prop->float_state.min, prop->float_state.max);
    if (current_val != save_val)
        gm_prop_set_float(prop, current_val);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", prop->desc);
}

static void
draw_vec3_property(Data *data,
                   struct gm_ui_properties *props,
                   struct gm_ui_property *prop,
                   const char *readable_name)
{
    if (prop->read_only) {
        ImGui::LabelText(readable_name, "%.3f,%.3f,%.3f",
                         //prop->vec3_state.components[0],
                         prop->vec3_state.ptr[0],
                         //prop->vec3_state.components[1],
                         prop->vec3_state.ptr[1],
                         //prop->vec3_state.components[2],
                         prop->vec3_state.ptr[2]);
    } // else TODO

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", prop->desc);
}

static void
draw_properties(Data *data,
                struct gm_ui_properties *props,
                std::unordered_map<std::string, void*> &overrides)
{
    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        if (prop->read_only) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        }

        const char *prop_name = prop->name;
        char readable_name[64];
        make_readable_name(prop_name, readable_name, sizeof(readable_name));

        if (overrides.find(prop_name) != overrides.end())
        {
            control_prop_draw_callback_t callback =
                (control_prop_draw_callback_t)overrides.at(prop_name);

            // We might associate a property with a NULL callback as a way
            // of skipping/hiding the property (maybe because we have manually
            // drawn a widget somewhere else).
            if (callback)
                callback(data, props, prop, readable_name);
        }
        else
        {
            switch (prop->type) {
            case GM_PROPERTY_INT:
                draw_int_property(data, props, prop, readable_name);
                break;
            case GM_PROPERTY_ENUM:
                draw_enum_property(data, props, prop, readable_name);
                break;
            case GM_PROPERTY_BOOL:
                draw_bool_property(data, props, prop, readable_name);
                break;
            case GM_PROPERTY_SWITCH:
                {
                    if (i && props->properties[i-1].type == GM_PROPERTY_SWITCH) {
                        ImGui::SameLine();
                    }
                    if (ImGui::Button(readable_name)) {
                        gm_prop_set_switch(prop);
                    }
                }
                break;
            case GM_PROPERTY_FLOAT:
                draw_float_property(data, props, prop, readable_name);
                break;
            case GM_PROPERTY_FLOAT_VEC3:
                draw_vec3_property(data, props, prop, readable_name);
                break;
            // FIXME: Handle GM_PROPERTY_STRING
            }
        }

        if (prop->read_only) {
            ImGui::PopStyleVar();
            ImGui::PopItemFlag();
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

static struct gm_ui_property *
find_prop(struct gm_ui_properties *props, const char *name)
{
    for (int p = 0; p < props->n_properties; ++p) {
        struct gm_ui_property *prop = &props->properties[p];

        if (prop->read_only)
            continue;

        if (strcmp(name, prop->name) == 0)
            return prop;
    }

    return NULL;
}

static GLuint
gen_ar_video_texture(Data *data)
{
    GLuint ar_video_tex;

    glGenTextures(1, &ar_video_tex);

    GLenum target = GL_TEXTURE_2D;
    if (gm_device_get_type(data->active_device) == GM_DEVICE_TANGO) {
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
update_ar_video_queue_len(Data *data, int len)
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
get_next_ar_video_tex(Data *data)
{
    if (data->ar_video_queue_len < 1) {
        update_ar_video_queue_len(data, 1);
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
get_oldest_ar_video_tex(Data *data)
{
    if (data->ar_video_queue.size() < data->ar_video_queue_len) {
        return data->ar_video_queue[0];
    } else {
        int oldest = (data->ar_video_queue_pos + 1) % data->ar_video_queue_len;
        return data->ar_video_queue[oldest];
    }
}

static void
update_target_skeleton_wireframe_gl_bos(Data *data,
                                        const struct gm_skeleton *ref_skeleton)
{
    if (!data->target ||
        gm_target_get_n_frames(data->target) == 0) {
        return;
    }

    const struct gm_skeleton *skeleton = gm_target_get_skeleton(data->target);
    struct gm_skeleton *resized_skeleton = NULL;
    if (ref_skeleton && data->target_resize) {
        resized_skeleton = gm_skeleton_resize(data->ctx,
                                              skeleton, ref_skeleton, 0);
        if (resized_skeleton)
            skeleton = (const struct gm_skeleton *)resized_skeleton;
    }

    data->target_skel_gl.n_joints = gm_skeleton_get_n_joints(skeleton);

    XYZRGBA colored_joints[data->target_skel_gl.n_joints];
    for (int i = 0; i < data->target_skel_gl.n_joints; i++) {
        const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);
        if (joint) {
            colored_joints[i].x = joint->x;
            colored_joints[i].y = joint->y;
            colored_joints[i].z = joint->z;
            colored_joints[i].rgba = LOOP_INDEX(joint_palette, i);
        } else {
            /* TODO: do something smarter... */
            colored_joints[i].x = 0;
            colored_joints[i].y = 0;
            colored_joints[i].z = 0;
            colored_joints[i].rgba = LOOP_INDEX(joint_palette, i);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, data->target_skel_gl.joints_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(XYZRGBA) * data->target_skel_gl.n_joints,
                 colored_joints, GL_DYNAMIC_DRAW);

    data->target_skel_gl.n_bones = gm_skeleton_get_n_bones(skeleton);
    XYZRGBA colored_bones[data->target_skel_gl.n_bones * 2];
    for (int b = 0; b < data->target_skel_gl.n_bones; ++b) {
        const struct gm_bone *bone = gm_skeleton_get_bone(skeleton, b);
        colored_bones[b*2] = colored_joints[gm_bone_get_head(bone)];
        colored_bones[b*2+1] = colored_joints[gm_bone_get_tail(bone)];
    }

    glBindBuffer(GL_ARRAY_BUFFER, data->target_skel_gl.bones_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(XYZRGBA) * data->target_skel_gl.n_bones * 2,
                 colored_bones, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (resized_skeleton) {
        gm_skeleton_free(resized_skeleton);
    }
}

static bool
update_skeleton_wireframe_gl_bos(Data *data, uint64_t timestamp)
{
    if (!data->latest_tracking) {
        return false;
    }

    /*
     * Update labelled point cloud
     */
    struct gm_prediction *prediction =
        gm_context_get_prediction(data->ctx, timestamp);
    if (!prediction) {
        return false;
    }
    const struct gm_skeleton *skeleton = gm_prediction_get_skeleton(prediction);

    // TODO: Take confidence into account to decide whether or not to show
    //       a particular joint position.
    data->skel_gl.n_joints = gm_skeleton_get_n_joints(skeleton);

    // Reformat and copy over joint data
    XYZRGBA colored_joints[data->skel_gl.n_joints];
    for (int i = 0; i < data->skel_gl.n_joints; i++) {
        const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);
        if (joint) {
            colored_joints[i].x = joint->x;
            colored_joints[i].y = joint->y;
            colored_joints[i].z = joint->z;
            colored_joints[i].rgba = LOOP_INDEX(joint_palette, i);
        } else {
            /* TODO: do something smarter... */
            colored_joints[i].x = 0;
            colored_joints[i].y = 0;
            colored_joints[i].z = 0;
            colored_joints[i].rgba = LOOP_INDEX(joint_palette, i);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, data->skel_gl.joints_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(XYZRGBA) * data->skel_gl.n_joints,
                 colored_joints, GL_DYNAMIC_DRAW);

    // Reformat and copy over bone data
    data->skel_gl.n_bones = gm_skeleton_get_n_bones(skeleton);
    XYZRGBA colored_bones[data->skel_gl.n_bones * 2];
    for (int b = 0; b < data->skel_gl.n_bones; ++b) {
        const struct gm_bone *bone = gm_skeleton_get_bone(skeleton, b);
        colored_bones[b*2] = colored_joints[gm_bone_get_head(bone)];
        colored_bones[b*2+1] = colored_joints[gm_bone_get_tail(bone)];

        if (!data->target) {
            continue;
        }

        // Colourise bone depending on how close it is to the test target
        float intensity = gm_target_get_error(data->target, bone);
        uint8_t red = (uint8_t)(intensity * 255.f);
        uint8_t green = (uint8_t)((1.f-intensity) * 255.f);
        colored_bones[b*2].rgba =
            (0xFF)|(green<<16)|(red<<24);
        colored_bones[b*2+1].rgba = colored_bones[b*2].rgba;
    }
    glBindBuffer(GL_ARRAY_BUFFER, data->skel_gl.bones_bo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(XYZRGBA) * data->skel_gl.n_bones * 2,
                 colored_bones, GL_DYNAMIC_DRAW);

    // Clean-up
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Update target skeleton
    if (data->target) {
        if (data->target_progress &&
            gm_target_get_cumulative_error(data->target, skeleton) <=
            data->target_error) {
            unsigned int frame = gm_target_get_frame(data->target);
            if (frame < gm_target_get_n_frames(data->target) - 1) {
                gm_target_set_frame(data->target, frame + 1);
            } else {
                gm_target_set_frame(data->target, 0);
            }
        }
        update_target_skeleton_wireframe_gl_bos(data, skeleton);
    }

    gm_prediction_unref(prediction);

    return true;
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
    deinit_device_opengl(data);
}

static void
draw_target_controls(Data *data)
{
    ImGui::Combo("Skeleton target",
                 &data->selected_target,
                 data->target_names.data(),
                 data->target_names.size());

    if (ImGui::Button(data->target ? "Unload###target_load" :
                                     "Load###target_load")) {
        if (data->target) {
            gm_target_free(data->target);
            data->target = NULL;
        } else {
            char *err = NULL;
            char path_tmp[PATH_MAX];
            snprintf(path_tmp, sizeof(path_tmp),
                     "Targets/%s",
                     data->targets.at(data->selected_target));
            data->target =
                gm_target_new_from_index(data->ctx, data->log, path_tmp, &err);
            if (!data->target) {
                gm_error(data->log, "Failed to load target: %s", err);
                free(err);
            } else {
                gm_info(data->log, "Target loaded with %u frames",
                        gm_target_get_n_frames(data->target));
            }
        }
    }

    if (!data->target) {
        return;
    }

    ImGui::SliderFloat("Error target", &data->target_error, 0.f, 1.f);
    if (ImGui::Button("<<###target_prev")) {
        unsigned int frame = gm_target_get_frame(data->target);
        if (frame > 0) {
            gm_target_set_frame(data->target, frame - 1);
        } else {
            gm_target_set_frame(data->target,
                                gm_target_get_n_frames(data->target) - 1);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(data->target_progress ? "||###target_pause" :
                                              ">###target_play")) {
        data->target_progress = !data->target_progress;
    }
    ImGui::SameLine();
    if (ImGui::Button(">>###target_next")) {
        unsigned int frame = gm_target_get_frame(data->target);
        if (frame < gm_target_get_n_frames(data->target) - 1) {
            gm_target_set_frame(data->target, frame + 1);
        } else {
            gm_target_set_frame(data->target, 0);
        }
    }

    ImGui::Checkbox("Resize target skeleton", &data->target_resize);
}

static void
draw_playback_controls(Data *data)
{
    if (ImGui::Button(data->recording ? "Stop" : "Record")) {
        if (data->recording) {
            gm_recording_close(data->recording);
            data->recording = NULL;
            index_recordings(data);
        } else if (!data->playback_device) {
            const char *rel_path = NULL;
            bool overwrite = false;
            if (data->overwrite_recording && data->recordings.size()) {
                rel_path = data->recordings.at(data->selected_playback_recording);
                overwrite = true;
            }

            data->recording = gm_recording_init(data->log,
                                                data->recording_device,
                                                glimpse_recordings_path,
                                                rel_path,
                                                overwrite);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(data->playback_device ?
                      "Unload###load_record" : "Load###load_record") &&
        !data->recording)
    {
        if (data->playback_device) {
            viewer_close_playback_device(data);

            // Wake up the recording device again
            handle_device_ready(data, data->recording_device);
        } else if (data->recordings.size()) {
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

            char idx_path[1024];
            snprintf(idx_path, 1024, "%s/%s",
                     glimpse_recordings_path,
                     data->recordings.at(data->selected_playback_recording));
            config.recording.path = dirname(idx_path);

            char *open_err = NULL;
            data->playback_device = gm_device_open(data->log, &config, &open_err);

            if (data->playback_device) {
                gm_device_set_event_callback(data->playback_device,
                                             on_device_event_cb, data);
                data->active_device = data->playback_device;
                deinit_device_opengl(data);

                gm_device_commit_config(data->playback_device, NULL);
            } else {
                gm_error(data->log, "Failed to start recording playback: %s",
                         open_err);
                free(open_err);
                // Wake up the recording device again
                handle_device_ready(data, data->recording_device);
            }
        }
    }

    ImGui::Spacing();

    if (!data->recording_names.empty()) {
        ImGui::Combo("Recording Path",
                     &data->selected_playback_recording,
                     data->recording_names.data(),
                     data->recording_names.size());
    }

    ImGui::Checkbox("Track while recording", &data->track_while_recording);

    if (data->playback_device) {
        struct gm_ui_properties *props =
            gm_device_get_ui_properties(data->playback_device);
        struct gm_ui_property *prop;

        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        prop = gm_props_lookup(props, "frame");
        draw_int_property(data, props, prop, "Frame");
        ImGui::PopStyleVar();
        ImGui::PopItemFlag();

        prop = gm_props_lookup(props, "max_frame");
        draw_int_property(data, props, prop, "Max Frame");

        prop = gm_props_lookup(props, "loop");
        draw_bool_property(data, props, prop, "Loop");
        prop = gm_props_lookup(props, "frame_skip");
        draw_bool_property(data, props, prop, "Frame Skip");
        prop = gm_props_lookup(props, "frame_throttle");
        draw_bool_property(data, props, prop, "Frame Throttle");

        prop = gm_props_lookup(props, "<<");
        if (ImGui::Button("<<")) {
            gm_prop_set_switch(prop);
        }
        ImGui::SameLine();
        prop = gm_props_lookup(props, "||>");
        if (ImGui::Button("||>")) {
            gm_prop_set_switch(prop);
        }
        prop = gm_props_lookup(props, ">>");
        ImGui::SameLine();
        if (ImGui::Button(">>")) {
            gm_prop_set_switch(prop);
        }
    }
}

static bool
draw_image_in_bounds(Data *data,
                     GLuint image_tex,
                     int image_width, int image_height, // only for aspect ratio
                     int bounds_width, int bounds_height,
                     enum gm_rotation rotation)
{
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
        std::swap(image_width, image_height);
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
        std::swap(image_width, image_height);
        break;
    }

    ImVec2 area_size(bounds_width, bounds_height);
    adjust_aspect(area_size, image_width, image_height);

    ImVec2 cur = ImGui::GetCursorScreenPos();

    ImGui::BeginGroup();
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->PushTextureID((void *)(intptr_t)image_tex);
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
    ImGui::EndGroup();

    return true;
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

    ImVec2 area_size = ImGui::GetContentRegionAvail();
    draw_image_in_bounds(data,
                         tex,
                         aspect_width, aspect_height,
                         area_size.x, area_size.y,
                         rotation);

    return focused;
}


static bool
draw_controls(Data *data, int x, int y, int width, int height, bool disabled)
{
    struct gm_ui_properties *ctx_props =
        gm_context_get_ui_properties(data->ctx);

    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    ImGui::SetNextWindowContentSize(ImVec2(width - ImGui::GetStyle().ScrollbarSize, 0));
    ImGui::Begin("Controls", NULL,
                 ImGuiWindowFlags_NoTitleBar|
                 ImGuiWindowFlags_NoResize|
                 ImGuiWindowFlags_NoMove|
                 ImGuiWindowFlags_AlwaysVerticalScrollbar|
                 ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() * 0.5);
    //ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5);

    if (disabled) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    }

    bool focused = ImGui::IsWindowFocused();

    ImGui::TextDisabled("Viewer properties...");
    ImGui::Separator();
    ImGui::Spacing();

    bool current_ar_mode = data->realtime_ar_mode;
    ImGui::Checkbox("Real-time AR Mode", &data->realtime_ar_mode);
    if (data->realtime_ar_mode != current_ar_mode)
    {
        if (data->realtime_ar_mode) {
            // Make sure to disable the debug cloud in real-time AR mode since it
            // may be costly to create.
            //
            // Note: We don't have to explicitly disable most debug views because
            // we only do work when we pull the data from the context, but that's
            // not the case for the cloud debug view.
            gm_prop_set_enum(find_prop(ctx_props, "cloud_mode"), 0);
        } else {
            gm_prop_set_enum(find_prop(ctx_props, "cloud_mode"), 1);
        }
    }

    ImGui::Checkbox("Show skeleton", &data->show_skeleton);
    ImGui::Checkbox("Show view camera controls", &data->show_view_cam_controls);
    ImGui::Checkbox("Show profiler", &data->show_profiler);
    ImGui::Checkbox("Show joint summary", &data->show_joint_summary);

    int queue_len = data->ar_video_queue_len;
    ImGui::SliderInt("AR video queue len", &queue_len, 1, 30);
    if (data->ar_video_queue_len != queue_len) {
        update_ar_video_queue_len(data, queue_len);
    }

    static const char *stage_stat_modes[] = {
        "Aggregated per-frame average",
        "Aggregated per-frame median",
        "Aggregated per-run average",
        "Aggregated per-run median",
        "Latest per-frame",
        "Latest per-run average",
        "Latest per-run median",
    };
    ImGui::Combo("Stage stats mode", &data->stage_stats_mode, stage_stat_modes,
                 ARRAY_LEN(stage_stat_modes));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", "Affects the way that the per-stage timing bar-graphs are updated");


    ImGui::Checkbox("Overwrite recording", &data->overwrite_recording);
    ImGui::SliderInt("Prediction delay", &data->prediction_delay,
                     0, 1000000000);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Playback controls...");
    ImGui::Separator();
    ImGui::Spacing();

    draw_playback_controls(data);

    if (!data->targets.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Target controls...");
        ImGui::Separator();
        ImGui::Spacing();

        draw_target_controls(data);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Device properties...");
    ImGui::Separator();
    ImGui::Spacing();

    draw_properties(data,
                    gm_device_get_ui_properties(data->active_device),
                    data->dev_control_overrides);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Mo-Cap properties...");
    ImGui::Separator();
    ImGui::Spacing();

    draw_properties(data,
                    ctx_props,
                    data->ctx_control_overrides);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Tracking Pipeline...");
    ImGui::Separator();
    ImGui::Spacing();

    uint64_t avg_frame_duration_ns = gm_context_get_average_frame_duration(data->ctx);
    uint64_t tracking_duration_ns = 0;
    if (data->latest_tracking)
        tracking_duration_ns = gm_tracking_get_duration(data->latest_tracking);

    int n_stages = gm_context_get_n_stages(data->ctx);
    for (int i = 0; i < n_stages; i++) {
        const char *stage_name = gm_context_get_stage_name(data->ctx, i);
        struct gm_ui_properties *stage_props =
            gm_context_get_stage_ui_properties(data->ctx, i);
        bool show_props = false;

        char readable_stage_name[64];
        make_readable_name(stage_name,
                           readable_stage_name,
                           sizeof(readable_stage_name));

        if (stage_props && stage_props->n_properties) {
            char stage_label[64];
            xsnprintf(stage_label, sizeof(stage_label),
                      "%sStage: %s###%s",
                      i == data->current_stage ? "* " : "",
                      readable_stage_name,
                      stage_name);

            show_props = ImGui::CollapsingHeader(stage_label);
        } else {
            ImGui::TextDisabled("%sStage: %s",
                                i == data->current_stage ? "* " : "",
                                readable_stage_name);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", gm_context_get_stage_description(data->ctx, i));
        }
        if (ImGui::IsItemClicked()) {
            data->current_stage = i;
            gm_prop_set_enum(find_prop(ctx_props, "debug_stage"), i);
        }

        struct stage_textures &stage_textures = data->stage_textures[i];
        int n_images = gm_context_get_stage_n_images(data->ctx, i);
        bool any_created_images = false;

        for (int n = 0; n < n_images; n++) {
            if (stage_textures.images[n].gl_tex) {
                any_created_images = true;
                break;
            }
        }

        if (any_created_images) {
            int max_width = ImGui::GetContentRegionAvailWidth();
            int w = max_width / n_images;
            int h = height / 5;

            int save_x = ImGui::GetCursorPosX();

            for (int n = 0; n < n_images; n++) {
                struct debug_image &debug_image = stage_textures.images[n];

                if (debug_image.gl_tex) {
                    GLuint view_tex = debug_image.gl_tex;
                    int view_tex_width = debug_image.width;
                    int view_tex_height = debug_image.height;

                    draw_image_in_bounds(data,
                                         view_tex,
                                         view_tex_width, view_tex_height, // aspect ratio of texture
                                         w, h,
                                         GM_ROTATION_0);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + w);
                }
            }
            ImGui::SetCursorPosX(save_x);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + h);
        }

        if (data->latest_tracking) {
            uint64_t ref_duration_ns = 0;
            uint64_t stage_duration_ns = 0;

            switch (data->stage_stats_mode)
            {
            case 0: // "Aggregated per-frame average",
                ref_duration_ns = avg_frame_duration_ns;
                stage_duration_ns =
                    gm_context_get_stage_frame_duration_avg(data->ctx, i);
                break;
            case 1: // "Aggregated per-frame median",
                ref_duration_ns = avg_frame_duration_ns;
                stage_duration_ns =
                    gm_context_get_stage_frame_duration_median(data->ctx, i);
                break;
            case 2: // "Aggregated per-run average",
                ref_duration_ns = avg_frame_duration_ns;
                stage_duration_ns =
                    gm_context_get_stage_run_duration_avg(data->ctx, i);
                break;
            case 3: // "Aggregated per-run median",
                ref_duration_ns = avg_frame_duration_ns;
                stage_duration_ns =
                    gm_context_get_stage_run_duration_median(data->ctx, i);
                break;
            case 4: // "Latest per-frame",
                ref_duration_ns = tracking_duration_ns;
                stage_duration_ns =
                    gm_tracking_get_stage_duration(data->latest_tracking, i);
                break;
            case 5: // "Latest per-run average",
                ref_duration_ns = tracking_duration_ns;
                stage_duration_ns =
                    gm_tracking_get_stage_run_duration_avg(data->latest_tracking, i);
                break;
            case 6: // "Latest per-run median",
                ref_duration_ns = tracking_duration_ns;
                stage_duration_ns =
                    gm_tracking_get_stage_run_duration_median(data->latest_tracking, i);
                break;
            }
            float fraction = (double)stage_duration_ns / ref_duration_ns;

            char duration_s16[16];
            format_duration_s16(stage_duration_ns, duration_s16);
            char buf[32];
            xsnprintf(buf, sizeof(buf), "%3.f%%/%s", (fraction * 100.f), duration_s16);

            ImGui::ProgressBar(fraction, ImVec2(-1.0f, 0.0f), buf);
        }

        if (show_props) {
            draw_properties(data,
                            stage_props,
                            data->stage_control_overrides);
        }

        ImGui::Spacing();
        ImGui::Separator();
    }


    if (ImGui::Button("Save config")) {
        JSON_Value *props_object = json_value_init_object();
        gm_props_to_json(data->log, ctx_props, props_object);

        JSON_Value *stages = json_value_init_object();
        json_object_set_value(json_object(props_object), "_stages", stages);
        for (int i = 0; i < n_stages; i++) {
            const char *stage_name = gm_context_get_stage_name(data->ctx, i);
            struct gm_ui_properties *stage_props =
                gm_context_get_stage_ui_properties(data->ctx, i);
            JSON_Value *stage_props_object = json_value_init_object();

            gm_props_to_json(data->log, stage_props, stage_props_object);

            json_object_set_value(json_object(stages), stage_name,
                                  stage_props_object);
        }

        char *json = json_serialize_to_string_pretty(props_object);
        json_value_free(props_object);
        props_object = NULL;

        const char *assets_root = gm_get_assets_root();
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

    ImGui::SameLine();
    if (ImGui::Button("Save skeleton") && data->latest_tracking &&
        gm_tracking_has_skeleton(data->latest_tracking)) {
        char output_name[1024];
        snprintf(output_name, 1024, "%06" PRIu64 ".json",
                 gm_tracking_get_timestamp(data->latest_tracking));
        const struct gm_skeleton *skeleton =
            gm_tracking_get_skeleton(data->latest_tracking);
        if (!gm_skeleton_save(skeleton, output_name)) {
            gm_error(data->log, "Error writing skeleton '%s'", output_name);
        }
    }

    if (disabled) {
        ImGui::PopItemFlag();
    }
    ImGui::PopItemWidth();

    ImGui::End();

    return focused;
}

static void
draw_joint_summary(Data *data)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

    ImGui::Begin("Joint details");

    float label_width = ImGui::GetContentRegionAvailWidth() * 0.275;
    float col_width = (ImGui::GetContentRegionAvailWidth() -
                       label_width) / 2.f;

    ImGui::Columns(3);
    ImGui::SetColumnWidth(0, label_width);
    ImGui::SetColumnWidth(1, col_width);
    ImGui::NextColumn();
    ImGui::TextDisabled("Raw");
    ImGui::NextColumn();
    ImGui::TextDisabled("Corrected");
    ImGui::NextColumn();

    if (data->latest_tracking &&
        gm_tracking_has_skeleton(data->latest_tracking))
    {
        const struct gm_skeleton *skel =
            gm_tracking_get_raw_skeleton(data->latest_tracking);
        const struct gm_skeleton *skel_ec =
            gm_tracking_get_skeleton(data->latest_tracking);

        for (int i = 0; i < gm_skeleton_get_n_joints(skel_ec); ++i) {
            const struct gm_joint *joint =
                gm_skeleton_get_joint(skel, i);
            const struct gm_joint *joint_ec =
                gm_skeleton_get_joint(skel_ec, i);

            ImGui::Columns(3);
            ImGui::SetColumnWidth(0, label_width);
            ImGui::SetColumnWidth(1, col_width);

            ImGui::TextDisabled("%s", gm_context_get_joint_name(data->ctx, i));
            ImGui::NextColumn();

            if (!joint || !joint_ec) {
                ImGui::NextColumn();
                ImGui::NextColumn();
                continue;
            }

            bool corrected = (joint->x != joint_ec->x ||
                              joint->y != joint_ec->y ||
                              joint->z != joint_ec->z);

            if (!corrected) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            }

            float joint_coords[3];
            joint_coords[0] = joint->x;
            joint_coords[1] = joint->y;
            joint_coords[2] = joint->z;
            ImGui::PushItemWidth(-1);
            ImGui::InputFloat3("", joint_coords, "%.2f",
                               ImGuiInputTextFlags_ReadOnly);
            ImGui::PopItemWidth();

            ImGui::NextColumn();

            if (corrected) {
                joint_coords[0] = joint_ec->x;
                joint_coords[1] = joint_ec->y;
                joint_coords[2] = joint_ec->z;
                ImGui::PushItemWidth(-1);
                ImGui::InputFloat3("", joint_coords, "%.2f",
                                   ImGuiInputTextFlags_ReadOnly);
                ImGui::PopItemWidth();
            } else {
                ImGui::PopStyleColor();
            }

            ImGui::NextColumn();
        }
    }
    ImGui::End();

    ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding
    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
}

static void
draw_skeleton_wireframe(Data *data, GLSkeleton *skel,
                        glm::mat4 mvp,
                        float pt_size)
{
    glUseProgram(data->cloud_program);

    // Set projection transform
    glUniformMatrix4fv(data->cloud_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    // Enable vertex arrays for drawing joints/bones
    glEnableVertexAttribArray(data->cloud_attr_pos);
    glEnableVertexAttribArray(data->cloud_attr_col);

    glBindBuffer(GL_ARRAY_BUFFER, skel->bones_bo);

    glVertexAttribPointer(data->cloud_attr_pos, 3, GL_FLOAT,
                          GL_FALSE, sizeof(XYZRGBA), nullptr);
    glVertexAttribPointer(data->cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                          GL_TRUE, sizeof(XYZRGBA),
                          (void *)offsetof(XYZRGBA, rgba));

    glDrawArrays(GL_LINES, 0, skel->n_bones * 2);

    glUniform1f(data->cloud_uniform_pt_size, pt_size * 3.f);

    glBindBuffer(GL_ARRAY_BUFFER, skel->joints_bo);

    glVertexAttribPointer(data->cloud_attr_pos, 3, GL_FLOAT,
                          GL_FALSE, sizeof(XYZRGBA), nullptr);
    glVertexAttribPointer(data->cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                          GL_TRUE, sizeof(XYZRGBA),
                          (void *)offsetof(XYZRGBA, rgba));

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, skel->n_joints);
    glDisable(GL_PROGRAM_POINT_SIZE);

    glDisableVertexAttribArray(data->cloud_attr_pos);
    glDisableVertexAttribArray(data->cloud_attr_col);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
}

static void
draw_debug_lines(Data *data, glm::mat4 mvp)
{
    if (!data->n_lines)
        return;

    glUseProgram(data->cloud_program);

    glUniformMatrix4fv(data->cloud_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    glEnableVertexAttribArray(data->cloud_attr_pos);
    glEnableVertexAttribArray(data->cloud_attr_col);

    glBindBuffer(GL_ARRAY_BUFFER, data->lines_bo);

    glVertexAttribPointer(data->cloud_attr_pos, 3, GL_FLOAT,
                          GL_FALSE, sizeof(struct gm_point_rgba), nullptr);
    glVertexAttribPointer(data->cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                          GL_TRUE, sizeof(struct gm_point_rgba),
                          (void *)offsetof(struct gm_point_rgba, rgba));

    glDrawArrays(GL_LINES, 0, data->n_lines * 2);

    glDisableVertexAttribArray(data->cloud_attr_pos);
    glDisableVertexAttribArray(data->cloud_attr_col);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
}

static void
draw_tracking_scene_to_texture(Data *data,
                               struct gm_tracking *tracking,
                               ImVec2 win_size, ImVec2 uiScale)
{
    GLint saved_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &saved_fbo);

    // Ensure the framebuffer texture is valid
    if (!data->cloud_fbo_valid) {
        int width = win_size.x * uiScale.x;
        int height = win_size.y * uiScale.y;

        // Generate textures
        glBindTexture(GL_TEXTURE_2D, data->cloud_fbo_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        // Bind colour/depth to point-cloud fbo
        glBindFramebuffer(GL_FRAMEBUFFER, data->cloud_fbo);

        glBindRenderbuffer(GL_RENDERBUFFER, data->cloud_depth_renderbuf);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, data->cloud_depth_renderbuf);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               data->cloud_fbo_tex, 0);

        GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);

        gm_assert(data->log,
                  (glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
                   GL_FRAMEBUFFER_COMPLETE),
                  "Incomplete framebuffer\n");

        data->cloud_fbo_valid = true;
    }

    if (data->cloud_bo &&
        data->cloud_intrinsics.width &&
        data->cloud_intrinsics.height)
    {
        struct gm_intrinsics *debug_intrinsics =
            &data->cloud_intrinsics;
        glm::mat4 proj = intrinsics_to_zoomed_project_matrix(debug_intrinsics,
                                                             0.01f, 10, // near, far
                                                             data->view_zoom);

        /* NB: we're rendering to an intermediate 2D texture with a bottom left
         * origin of (0,0).
         *
         * By default ImGui::Image() assumes you're mapping uv0=(0,0) and
         * uv1=(1,1) to the top-left and bottom-right of a quad, respectively.
         *
         * We flip Y here so our render-to-texture results will be the right
         * after the texture is sampled (with a second flips) via ImGui::Image().
         *
         * We don't want to mess with the uv coordinates we pass to
         * ImGui::Image() since it would probably be better to just be
         * consistent with what ImGui expects by default (notably convenient
         * for cases where images are loaded in scanline order from memory,
         * which technically leaves them 'upside down' by GL conventions and
         * typically need flipping too.)
         */
        glm::mat4 mvp = glm::scale(proj, glm::vec3(1.0, -1.0, -1.0));
        mvp = glm::translate(mvp, data->focal_point);
        mvp = glm::rotate(mvp, data->camera_rot_yx[0], glm::vec3(0.0, 1.0, 0.0));
        mvp = glm::rotate(mvp, data->camera_rot_yx[1], glm::vec3(1.0, 0.0, 0.0));
        mvp = glm::translate(mvp, -data->focal_point);

        glBindFramebuffer(GL_FRAMEBUFFER, data->cloud_fbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, win_size.x * uiScale.x, win_size.y * uiScale.y);

        glUseProgram(data->cloud_program);
        glUniformMatrix4fv(data->cloud_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

        float pt_size = ceilf((win_size.x * uiScale.x * data->view_zoom) /
                              debug_intrinsics->width);
        glUniform1f(data->cloud_uniform_pt_size, pt_size);

        glBindBuffer(GL_ARRAY_BUFFER, data->cloud_bo);
        if (data->cloud_attr_pos != -1) {
            glEnableVertexAttribArray(data->cloud_attr_pos);
            glVertexAttribPointer(data->cloud_attr_pos, 3, GL_FLOAT, GL_FALSE,
                                  sizeof(struct gm_point_rgba), 0);
        }
        glEnableVertexAttribArray(data->cloud_attr_col);
        glVertexAttribPointer(data->cloud_attr_col, 4, GL_UNSIGNED_BYTE,
                              GL_TRUE,
                              sizeof(struct gm_point_rgba),
                              (void *)offsetof(struct gm_point_rgba, rgba));

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDepthFunc(GL_LESS);

        glDrawArrays(GL_POINTS, 0, data->n_cloud_points);

        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_DEPTH_TEST);

        glDisableVertexAttribArray(data->cloud_attr_pos);
        if (data->cloud_attr_pos != -1)
            glDisableVertexAttribArray(data->cloud_attr_col);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

        if (data->show_skeleton &&
            update_skeleton_wireframe_gl_bos(data,
                                             gm_tracking_get_timestamp(data->latest_tracking)))
        {
            if (data->target) {
                glm::mat4 mvp2 = glm::scale(mvp, glm::vec3(0.3f, 0.3f, 0.3f));
                mvp2 = glm::translate(mvp2, glm::vec3(-1.5f, 0.f, 2.0f));
                draw_skeleton_wireframe(data, &data->target_skel_gl, mvp2, pt_size);
            }
            draw_skeleton_wireframe(data, &data->skel_gl, mvp, pt_size);
        }

        draw_debug_lines(data, mvp);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, saved_fbo);
}

static bool
draw_cloud_visualisation(Data *data, ImVec2 &uiScale,
                         int x, int y, int width, int height)
{
    if (!data->latest_tracking)
        return false;

    const struct gm_intrinsics *depth_intrinsics =
        gm_tracking_get_depth_camera_intrinsics(data->latest_tracking);
    int depth_width = depth_intrinsics->width;
    int depth_height = depth_intrinsics->height;

    bool focused = draw_visualisation(data, x, y, width, height,
                                      depth_width, depth_height,
                                      "Cloud", 0, GM_ROTATION_0);

    ImVec2 win_size = ImGui::GetContentRegionMax();
    adjust_aspect(win_size, depth_width, depth_height);
    draw_tracking_scene_to_texture(data, data->latest_tracking, win_size, uiScale);

    ImGui::Image((void *)(intptr_t)data->cloud_fbo_tex, win_size);

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

static void
reset_view(Data *data)
{
    data->view_zoom = 1;
    data->focal_point = glm::vec3(0.0, 0.0, 2.5);
    data->camera_rot_yx[0] = 0;
    data->camera_rot_yx[1] = 0;
}

static void
draw_view_camera_controls(Data *data)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

    ImGui::Begin("Camera Controls");

    ImGui::SliderFloat("Zoom", &data->view_zoom, 0.1, 3);

    if (ImGui::Button("Reset")) {
        reset_view(data);
    }
    ImGui::End();

    ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding
    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
}
static void
draw_ui(Data *data)
{
    ProfileScopedSection(DrawIMGUI, ImGuiControl::Profiler::Dark);

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 uiScale = io.DisplayFramebufferScale;
    ImVec2 origin = io.DisplayVisibleMin;
    ImVec2 win_size = ImVec2(io.DisplayVisibleMax.x - io.DisplayVisibleMin.x,
                             io.DisplayVisibleMax.y - io.DisplayVisibleMin.y);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

    int main_x = origin.x;
    int main_y = origin.y;
    ImVec2 main_area_size = win_size;

    bool show_controls_button = false;
    int controls_x = origin.x;
    int controls_y = origin.y;
    int controls_width = main_area_size.x;
    int controls_height = main_area_size.y;

    if (data->realtime_ar_mode) {
        show_controls_button = true;
    }

    if (win_size.x < 1024 || win_size.y < 600) {
        show_controls_button = true;
    } else {
        controls_width = TOOLBAR_WIDTH;
        main_x += controls_width;
        main_area_size.x -= controls_width;
    }

    /* NB: we don't use imgui to render the video background while in
     * real-time mode
     */
    if (!data->realtime_ar_mode) {
        draw_cloud_visualisation(data,
                                 uiScale,
                                 main_x, main_y,
                                 main_area_size.x,
                                 main_area_size.y);
    }

    static bool show_controls = false;

    if (show_controls_button) {
        // Draw a view-picker at the top
        ImGui::SetNextWindowPos(origin);
        //ImGui::SetNextWindowContentSize(ImVec2(0, 0));
        //ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0),
        //                                    ImVec2(0, 0));
        //ImGui::SetNextWindowSizeConstraints(ImVec2(win_size.x, 0),
        //                                    ImVec2(win_size.x, win_size.y));
        ImGui::Begin("Controls Toggle", NULL,
                     ImGuiWindowFlags_NoTitleBar|
                     ImGuiWindowFlags_NoResize|
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize|
                     ImGuiWindowFlags_NoMove);
        /* XXX: assuming that "Controls" and "Video Buffer" are the first two
         * entries, we only want to expose these two options in
         * realtime_ar_mode, while we aren't uploading any other debug textures
         */
        if (ImGui::Button(show_controls ? "Close" : "Properties")) {
            show_controls = !show_controls;
        }

        controls_y = origin.y + ImGui::GetWindowHeight();
        controls_height -= ImGui::GetWindowHeight();

        ImGui::End();
    } else
        show_controls = true;

    if (show_controls) {
        draw_controls(data, controls_x, controls_y,
                      controls_width, controls_height,
                      false); // enabled
    }

    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowRounding

    if (data->show_profiler) {
        // Draw profiler window always-on-top
        ImGui::SetNextWindowPos(origin, ImGuiCond_Once);
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
        ProfileDrawUI();
    }

    if (data->show_joint_summary) {
        ImGui::SetNextWindowPos(origin, ImGuiCond_Once);
        draw_joint_summary(data);
    }

    if (data->show_view_cam_controls) {
        ImGui::SetNextWindowPos(origin, ImGuiCond_Once);
        draw_view_camera_controls(data);
    }

    //ImGui::ShowTestWindow();

    ImGui::Render();
}

static void
draw_ar_video(Data *data)
{
    if (!data->device_gl_initialized || data->last_video_frame == NULL)
        return;

    gm_assert(data->log, !!data->ctx, "draw_ar_video, NULL ctx");

    enum gm_rotation rotation = data->last_video_frame->camera_rotation;
    const struct gm_intrinsics *video_intrinsics =
        &data->last_video_frame->video_intrinsics;
    int video_width = video_intrinsics->width;
    int video_height = video_intrinsics->height;

    int aspect_width = video_width;
    int aspect_height = video_height;

    struct {
        float x, y, s, t;
    } xyst_verts[4] = {
        { -1,  1, 0, 0, }, //  0 -- 1
        {  1,  1, 1, 0, }, //  | \  |
        {  1, -1, 1, 1  }, //  |  \ |
        { -1, -1, 0, 1, }, //  3 -- 2
    };
    int n_verts = ARRAY_LEN(xyst_verts);

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
        std::swap(aspect_width, aspect_height);
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
        std::swap(aspect_width, aspect_height);
        break;
    }

    float display_aspect = data->win_width / (float)data->win_height;
    float video_aspect = aspect_width / (float)aspect_height;
    float aspect_x_scale = 1;
    float aspect_y_scale = 1;
    if (video_aspect > display_aspect) {
        // fit by scaling down y-axis of video
        float fit_height = (float)data->win_width / video_aspect;
        aspect_y_scale = fit_height / (float)data->win_height;
    } else {
        // fit by scaling x-axis of video
        float fit_width = video_aspect * data->win_height;
        aspect_x_scale = fit_width / (float)data->win_width;
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

    /* trivial enough to just do the transform on the cpu... */
    for (int i = 0; i < n_verts; i++) {
        xyst_verts[i].x *= aspect_x_scale;
        xyst_verts[i].y *= aspect_y_scale;
    }

    /* XXX: we could just cache buffers for each rotation */
    glBindBuffer(GL_ARRAY_BUFFER, data->video_quad_attrib_bo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n_verts,
                 xyst_verts, GL_STATIC_DRAW);

    glUseProgram(data->video_program);
    glBindBuffer(GL_ARRAY_BUFFER, data->video_quad_attrib_bo);

    glEnableVertexAttribArray(data->video_quad_attrib_pos);
    glVertexAttribPointer(data->video_quad_attrib_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)0);

    if (data->video_quad_attrib_tex_coords != -1) {
        glEnableVertexAttribArray(data->video_quad_attrib_tex_coords);
        glVertexAttribPointer(data->video_quad_attrib_tex_coords,
                              2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void *)8);
    }

    enum gm_device_type device_type = gm_device_get_type(data->active_device);
    GLenum target = GL_TEXTURE_2D;
    if (device_type == GM_DEVICE_TANGO)
        target = GL_TEXTURE_EXTERNAL_OES;
    GLuint ar_video_tex = get_oldest_ar_video_tex(data);
    glBindTexture(target, ar_video_tex);

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLE_FAN, 0, n_verts);
    gm_debug(data->log, "draw_video");
    glDepthMask(GL_TRUE);

    glBindTexture(target, 0);

    glDisableVertexAttribArray(data->video_quad_attrib_pos);
    if (data->video_quad_attrib_tex_coords != -1)
        glDisableVertexAttribArray(data->video_quad_attrib_tex_coords);

    glUseProgram(0);

    if (data->show_skeleton && data->latest_tracking) {
        struct gm_intrinsics rotated_intrinsics;

        gm_context_rotate_intrinsics(data->ctx,
                                     video_intrinsics,
                                     &rotated_intrinsics,
                                     rotation);

        float pt_size = ((float)data->win_width / 240.0f) * aspect_x_scale;
        glm::mat4 proj = intrinsics_to_zoomed_project_matrix(&rotated_intrinsics,
                                                             0.01f, 10, // near, far
                                                             data->view_zoom);
        glm::mat4 mvp = glm::scale(proj, glm::vec3(aspect_x_scale, aspect_y_scale, -1.0));

        if (update_skeleton_wireframe_gl_bos(data,
                                             data->last_video_frame->timestamp -
                                             data->prediction_delay))
        {
            if (data->target) {
                glm::mat4 mvp2 = glm::scale(mvp, glm::vec3(0.3f, 0.3f, 0.3f));
                mvp2 = glm::translate(mvp2, glm::vec3(-1.5f, 0.f, 2.0f));
                draw_skeleton_wireframe(data, &data->target_skel_gl, mvp2, pt_size);
            }
            draw_skeleton_wireframe(data, &data->skel_gl, mvp, pt_size);
        }
    }
}

/* If we've already requested gm_device for a frame then this won't submit
 * a request that downgrades the buffers_mask
 */
static void
request_device_frame(Data *data, uint64_t buffers_mask)
{
    uint64_t new_buffers_mask = data->pending_frame_buffers_mask | buffers_mask;

    if (data->pending_frame_buffers_mask != new_buffers_mask) {
        gm_device_request_frame(data->active_device, new_buffers_mask);
        data->pending_frame_buffers_mask = new_buffers_mask;
    }
}

static void
handle_device_frame_updates(Data *data)
{
    ProfileScopedSection(UpdatingDeviceFrame);
    bool upload_video_texture = false;

    if (!data->device_frame_ready)
        return;

    {
        ProfileScopedSection(GetLatestFrame);
        /* NB: gm_device_get_latest_frame will give us a _ref() */
        gm_frame *device_frame = gm_device_get_latest_frame(data->active_device);
        if (!device_frame) {
            return;
        }

        if (device_frame->depth) {
            if (data->last_depth_frame) {
                gm_frame_unref(data->last_depth_frame);
            }
            gm_frame_ref(device_frame);
            data->last_depth_frame = device_frame;
            data->pending_frame_buffers_mask &= ~GM_REQUEST_FRAME_DEPTH;
        }

        if (device_frame->video) {
            if (data->last_video_frame) {
                gm_frame_unref(data->last_video_frame);
            }
            gm_frame_ref(device_frame);
            data->last_video_frame = device_frame;
            data->pending_frame_buffers_mask &= ~GM_REQUEST_FRAME_VIDEO;
            upload_video_texture = true;
        }

        if (data->recording) {
            gm_recording_save_frame(data->recording, device_frame);
        }

        gm_frame_unref(device_frame);
    }

    if (data->context_needs_frame &&
        data->last_depth_frame && data->last_video_frame &&
        (data->track_while_recording || !data->recording))
    {
        ProfileScopedSection(FwdContextFrame);

        // Combine the two video/depth frames into a single frame for gm_context
        if (data->last_depth_frame != data->last_video_frame) {
            struct gm_frame *full_frame =
                gm_device_combine_frames(data->active_device,
                                         data->last_depth_frame,
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
         * Note: the buffers_mask may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, data->recording ?
                             (GM_REQUEST_FRAME_DEPTH | GM_REQUEST_FRAME_VIDEO) :
                             GM_REQUEST_FRAME_VIDEO);
    }

    enum gm_device_type device_type = gm_device_get_type(data->active_device);

    if (upload_video_texture && data->device_gl_initialized) {
        if (device_type != GM_DEVICE_TANGO) {
            const struct gm_intrinsics *video_intrinsics =
                &data->last_video_frame->video_intrinsics;
            int video_width = video_intrinsics->width;
            int video_height = video_intrinsics->height;

            ProfileScopedSection(UploadFrameTextures);

            /*
             * Update video from camera
             */
            GLuint ar_video_tex = get_next_ar_video_tex(data);
            glBindTexture(GL_TEXTURE_2D, ar_video_tex);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

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
            case GM_FORMAT_BGR_U8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                             video_width, video_height,
                             0, GL_BGR, GL_UNSIGNED_BYTE, video_front);
                break;

            case GM_FORMAT_RGBX_U8:
            case GM_FORMAT_RGBA_U8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                             video_width, video_height,
                             0, GL_RGBA, GL_UNSIGNED_BYTE, video_front);
                break;
            case GM_FORMAT_BGRX_U8:
            case GM_FORMAT_BGRA_U8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                             video_width, video_height,
                             0, GL_BGRA, GL_UNSIGNED_BYTE, video_front);
                break;

            case GM_FORMAT_UNKNOWN:
            case GM_FORMAT_Z_U16_MM:
            case GM_FORMAT_Z_F32_M:
            case GM_FORMAT_Z_F16_M:
            case GM_FORMAT_POINTS_XYZC_F32_M:
                gm_assert(data->log, 0, "Unexpected format for video buffer");
                break;
            }
        } else {
#ifdef USE_TANGO
            GLuint ar_video_tex = get_next_ar_video_tex(data);
            if (TangoService_updateTextureExternalOes(
                    TANGO_CAMERA_COLOR, ar_video_tex,
                    NULL /* ignore timestamp */) != TANGO_SUCCESS)
            {
                gm_warn(data->log, "Failed to update video frame via TangoService_updateTextureExternalOes");
            }
#endif
        }
    }
}

static void
upload_tracking_textures(Data *data)
{
    /* The tracking textures are all for debug purposes and we want to skip
     * the overhead of uploading them while in realtime_ar_mode
     */
    if (data->realtime_ar_mode)
        return;

    ProfileScopedSection(UploadTrackingBufs);

    int n_stages = gm_context_get_n_stages(data->ctx);
    gm_assert(data->log, n_stages == data->stage_textures.size(),
              "stage_textures size doesn't match number of stages");

    for (int i = 0; i < n_stages; i++) {
        int width;
        int height;
        int n_images = gm_context_get_stage_n_images(data->ctx, i);
        struct stage_textures &stage_textures = data->stage_textures[i];
        uint64_t stage_duration = gm_tracking_get_stage_duration(data->latest_tracking, i);

        for (int n = 0; n < n_images; n++) {
            struct debug_image &debug_image = stage_textures.images[n];
            uint8_t *rgb_data = NULL;

            if (stage_duration &&
                gm_tracking_create_stage_rgb_image(data->latest_tracking,
                                                   i,
                                                   n,
                                                   &width,
                                                   &height,
                                                   &rgb_data))
            {
                if (!debug_image.gl_tex) {
                    glGenTextures(1, &debug_image.gl_tex);
                    glBindTexture(GL_TEXTURE_2D, debug_image.gl_tex);

                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                }

                glBindTexture(GL_TEXTURE_2D, debug_image.gl_tex);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                             width, height,
                             0, GL_RGB, GL_UNSIGNED_BYTE, rgb_data);
                debug_image.width = width;
                debug_image.height = height;

                free(rgb_data);
                rgb_data = NULL;
            } else {
                glDeleteTextures(1, &debug_image.gl_tex);
                debug_image.gl_tex = 0;
                debug_image.width = 0;
                debug_image.height = 0;
            }
        }
    }

    int n_points = 0;
    struct gm_intrinsics debug_cloud_intrinsics = {};
    const struct gm_point_rgba *debug_points =
        gm_tracking_get_debug_point_cloud(data->latest_tracking,
                                          &n_points,
                                          &debug_cloud_intrinsics);
    if (n_points) {
        if (!data->cloud_bo)
            glGenBuffers(1, &data->cloud_bo);
        glBindBuffer(GL_ARRAY_BUFFER, data->cloud_bo);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(debug_points[0]) * n_points,
                     debug_points, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        data->n_cloud_points = n_points;
        data->cloud_intrinsics = debug_cloud_intrinsics;
    } else
        data->n_cloud_points = 0;

    int n_lines = 0;
    const struct gm_point_rgba *debug_lines =
        gm_tracking_get_debug_lines(data->latest_tracking, &n_lines);
    if (n_lines) {
        if (!data->lines_bo)
            glGenBuffers(1, &data->lines_bo);
        glBindBuffer(GL_ARRAY_BUFFER, data->lines_bo);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(debug_lines[0]) * n_lines * 2,
                     debug_lines, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        data->n_lines = n_lines;
    } else
        data->n_lines = 0;
}

static void
destroy_joints_recording(Data *data)
{
    if (data->joints_recording_val) {
        json_value_free(data->joints_recording_val);
        data->joints_recording_val = NULL;
        data->joints_recording = NULL;
    }
}

static void
start_joints_recording(Data *data)
{
    destroy_joints_recording(data);
    data->joints_recording_val = json_value_init_array();
    data->joints_recording = json_array(data->joints_recording_val);
}

static void
handle_context_tracking_updates(Data *data)
{
    ProfileScopedSection(UpdatingTracking);

    if (!data->tracking_ready)
        return;

    data->tracking_ready = false;

    if (data->latest_tracking) {
        gm_tracking_unref(data->latest_tracking);
    }

    data->latest_tracking = gm_context_get_latest_tracking(data->ctx);

    // When flushing the context, we can end up with notified tracking but
    // no tracking to pick up
    if (!data->latest_tracking) {
        return;
    }

    if (data->joints_recording) {
        const struct gm_skeleton *skeleton =
            gm_tracking_get_skeleton(data->latest_tracking);
        int n_joints = gm_skeleton_get_n_joints(skeleton);

        JSON_Value *joints_array_val = json_value_init_array();
        JSON_Array *joints_array = json_array(joints_array_val);
        for (int i = 0; i < n_joints; i++) {
            const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);
            JSON_Value *coord_val = json_value_init_array();
            JSON_Array *coord = json_array(coord_val);
            if (joint) {
                json_array_append_number(coord, joint->x);
                json_array_append_number(coord, joint->y);
                json_array_append_number(coord, joint->z);
            } else {
                /* TODO: do something smarter... */
                json_array_append_number(coord, 0);
                json_array_append_number(coord, 0);
                json_array_append_number(coord, 0);
            }

            json_array_append_value(joints_array, coord_val);
        }

        json_array_append_value(data->joints_recording, joints_array_val);

        int n_frames = json_array_get_count(data->joints_recording);
        if (n_frames >= data->requested_recording_len) {
            json_serialize_to_file_pretty(data->joints_recording_val,
                                          "glimpse-joints-recording.json");
            destroy_joints_recording(data);
        }
    }

    upload_tracking_textures(data);
}

static void
handle_device_ready(Data *data, struct gm_device *dev)
{
    gm_debug(data->log, "%s device ready\n",
            dev == data->playback_device ? "Playback" : "Default");

    init_viewer_opengl(data);
    init_device_opengl(data);

    int max_depth_pixels =
        gm_device_get_max_depth_pixels(dev);
    gm_context_set_max_depth_pixels(data->ctx, max_depth_pixels);

    int max_video_pixels =
        gm_device_get_max_video_pixels(dev);
    gm_context_set_max_video_pixels(data->ctx, max_video_pixels);

    /*gm_context_set_depth_to_video_camera_extrinsics(data->ctx,
      gm_device_get_depth_to_video_extrinsics(dev));*/

    char *catch_err = NULL;
    const char *device_config = "glimpse-device.json";
    if (!gm_device_load_config_asset(dev,
                                     device_config,
                                     &catch_err))
    {
        gm_warn(data->log, "Didn't open device config: %s", catch_err);
        free(catch_err);
        catch_err = NULL;
    }

    uint64_t old_reqs = data->pending_frame_buffers_mask;
    data->pending_frame_buffers_mask = 0;
    gm_device_start(dev);
    gm_context_enable(data->ctx);
    if (old_reqs) {
        request_device_frame(data, old_reqs);
    }

    if (data->requested_recording_len)
        start_joints_recording(data);
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

    /* To avoid redundant work; just in case there are multiple _TRACKING_READY
     * or _FRAME_READY notifications backed up then we squash them together and
     * handle after we've iterated all outstanding events...
     */
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
    data->cloud_fbo_valid = false;
}

static void
surface_destroyed_cb(GLFMDisplay *display)
{
    Data *data = (Data *)glfmGetUserData(display);
    gm_debug(data->log, "Surface destroyed");
    data->surface_created = false;
    data->cloud_fbo_valid = false;
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
            if (data->realtime_ar_mode)
                draw_ar_video(data);
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
            if (data->realtime_ar_mode)
                draw_ar_video(data);
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
    data->cloud_fbo_valid = false;
}

static void
on_key_input_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Data *data = (Data *)glfwGetWindowUserPointer(window);

    ImGui_ImplGlfwGLES3_KeyCallback(window, key, scancode, action, mods);

    if (action != GLFW_PRESS)
        return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(data->window, 1);
        break;
    }
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

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_cond_signal(&data->event_notify_cond);
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
#if 0
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
#endif

#if TARGET_OS_OSX == 1
    // In the forwards-compatible context, there's no default vertex array.
    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);
#endif
}

static void
init_viewer_opengl(Data *data)
{
    if (data->gl_initialized)
        return;

    static const char *cloud_vert_shader =
        GLSL_SHADER_VERSION
        "precision mediump float;\n"
        "uniform mat4 mvp;\n"
        "uniform float size;\n"
        "in vec3 pos;\n"
        "in vec4 color_in;\n"
        "out vec4 v_color;\n"
        "\n"
        "void main() {\n"
        "  gl_PointSize = size;\n"
        "  gl_Position =  mvp * vec4(pos.x, pos.y, pos.z, 1.0);\n"
        "  v_color = color_in;\n"
        "}\n";

    static const char *cloud_frag_shader =
        GLSL_SHADER_VERSION
        "precision mediump float;\n"
        "in vec4 v_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  color = v_color.abgr;\n"
        "}\n";

    data->cloud_program = gm_gl_create_program(data->log,
                                               cloud_vert_shader,
                                               cloud_frag_shader,
                                               NULL);

    glUseProgram(data->cloud_program);

    data->cloud_attr_pos = glGetAttribLocation(data->cloud_program, "pos");
    data->cloud_attr_col = glGetAttribLocation(data->cloud_program, "color_in");
    data->cloud_uniform_mvp = glGetUniformLocation(data->cloud_program, "mvp");
    data->cloud_uniform_pt_size = glGetUniformLocation(data->cloud_program, "size");

    glUseProgram(0);

    glGenBuffers(1, &data->lines_bo);
    glGenBuffers(1, &data->skel_gl.bones_bo);
    glGenBuffers(1, &data->skel_gl.joints_bo);

    glGenBuffers(1, &data->target_skel_gl.bones_bo);
    glGenBuffers(1, &data->target_skel_gl.joints_bo);

    int n_stages = gm_context_get_n_stages(data->ctx);
    data->stage_textures.resize(n_stages);

    for (int i = 0; i < n_stages; i++) {
        struct stage_textures &stage_textures = data->stage_textures[i];
        int n_images = gm_context_get_stage_n_images(data->ctx, i);

        gm_assert(data->log, n_images < MAX_IMAGES_PER_STAGE,
                  "Can't handle more than %d debug images per stage",
                  MAX_IMAGES_PER_STAGE);

        for (int n = 0; n < n_images; n++) {
            struct debug_image &debug_image = stage_textures.images[n];

            debug_image.gl_tex = 0;
            debug_image.width = 0;
            debug_image.height = 0;
        }
    }

    glGenFramebuffers(1, &data->cloud_fbo);
    glGenRenderbuffers(1, &data->cloud_depth_renderbuf);
    glGenTextures(1, &data->cloud_fbo_tex);
    glBindTexture(GL_TEXTURE_2D, data->cloud_fbo_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenBuffers(1, &data->video_quad_attrib_bo);

    data->gl_initialized = true;
}

static void
init_device_opengl(Data *data)
{
    if (data->device_gl_initialized)
        return;

    gm_assert(data->log, data->video_program == 0,
              "Spurious GL video_program while device_gl_initialized == false");

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

    if (gm_device_get_type(data->active_device) == GM_DEVICE_TANGO) {
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

    data->video_quad_attrib_pos =
        glGetAttribLocation(data->video_program, "pos");
    data->video_quad_attrib_tex_coords =
        glGetAttribLocation(data->video_program, "tex_coords_in");

    data->ar_video_tex_sampler = glGetUniformLocation(data->video_program, "tex_sampler");

    glUseProgram(data->video_program);
    glUniform1i(data->ar_video_tex_sampler, 0);
    glUseProgram(0);
    update_ar_video_queue_len(data, 6);

    // XXX: inconsistent that cloud_fbo is allocated in init_viewer_opengl
    data->cloud_fbo_valid = false;

    data->device_gl_initialized = true;
}

static void
deinit_device_opengl(Data *data)
{
    if (!data->device_gl_initialized)
        return;

    if (data->video_program) {
        glDeleteProgram(data->video_program);
        data->video_program = 0;

        data->video_quad_attrib_pos = 0;
        data->video_quad_attrib_tex_coords = 0;
        data->ar_video_tex_sampler = 0;
    }

    update_ar_video_queue_len(data, 0);

    int n_stages = gm_context_get_n_stages(data->ctx);
    for (int i = 0; i < n_stages; i++) {
        struct stage_textures &stage_textures = data->stage_textures[i];
        int n_images = gm_context_get_stage_n_images(data->ctx, i);

        for (int n = 0; n < n_images; n++) {
            struct debug_image &debug_image = stage_textures.images[n];

            if (debug_image.gl_tex)
                glDeleteTextures(1, &debug_image.gl_tex);
            debug_image.gl_tex = 0;
            debug_image.width = 0;
            debug_image.height = 0;
        }
    }

    // XXX: inconsistent that cloud_fbo is allocated in init_viewer_opengl
    data->cloud_fbo_valid = false;

    data->device_gl_initialized = false;
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
#if TARGET_OS_IOS == 1
            ios_log(msg);
#endif

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
            fflush(stdout);
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

    ImGui::CreateContext();
    ImGui_ImplGlfmGLES3_Init(display, true);

    // Quick hack to make scrollbars a bit more usable on small devices
    ImGui::GetStyle().ScrollbarSize *= 2;
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

#if TARGET_OS_OSX == 1
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  2) ;
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  0) ;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#endif

    data->window = glfwCreateWindow(1280, 720, "Glimpse Viewer", NULL, NULL);
    if (!data->window) {
        fprintf(stderr, "Failed to create window\n");
        exit(1);
    }


    glfwSetWindowUserPointer(data->window, data);

    glfwGetFramebufferSize(data->window, &data->win_width, &data->win_height);
    glfwSetFramebufferSizeCallback(data->window, on_window_fb_size_change_cb);

    glfwMakeContextCurrent(data->window);
    glfwSwapInterval(1);

    glfwSetErrorCallback(on_glfw_error_cb);

    ImGui::CreateContext();
    ImGui_ImplGlfwGLES3_Init(data->window, false /* don't install callbacks */);

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 ui_scale = io.DisplayFramebufferScale;
    ImGui::GetStyle().ScaleAllSizes(ui_scale.x);

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

    if (data->target) {
        gm_target_free(data->target);
    }

    gm_logger_destroy(data->log);

    delete data->events_front;
    delete data->events_back;

#ifdef USE_GLFW
    ImGui_ImplGlfwGLES3_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(data->window);
    glfwTerminate();
#endif

    delete data;

    ProfileShutdown();
}

static void
configure_recording_device(Data *data)
{
    gm_device_set_event_callback(data->recording_device, on_device_event_cb, data);
#ifdef __ANDROID__
    gm_device_attach_jvm(data->recording_device, android_jvm_singleton);
#endif
}

static void
viewer_init(Data *data)
{
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsClassic();

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
    if (n_frames_env)
        data->requested_recording_len = strtoull(n_frames_env, NULL, 10);

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    data->ctx = gm_context_new(data->log, NULL);

    gm_context_set_event_callback(data->ctx, on_event_cb, data);

    /* TODO: load config for viewer properties */
    data->prediction_delay = 250000000;

    struct gm_asset *config_asset =
        gm_asset_open(data->log,
                      "glimpse-config.json", GM_ASSET_MODE_BUFFER, &open_err);
    if (config_asset) {
        const char *buf = (const char *)gm_asset_get_buffer(config_asset);
        JSON_Value *json_config = json_parse_string(buf);
        gm_context_set_config(data->ctx, json_config);
        json_value_free(json_config);
        gm_asset_close(config_asset);
    } else {
        gm_warn(data->log, "Failed to open glimpse-config.json: %s", open_err);
        free(open_err);
    }

    struct gm_device_config config = {};
#ifdef USE_TANGO
    config.type = GM_DEVICE_TANGO;
#elif TARGET_OS_IOS == 1
    config.type = GM_DEVICE_AVF;
#else
    config.type = device_type_opt;
    char rec_path[1024];
    if (config.type == GM_DEVICE_RECORDING) {
        xsnprintf(rec_path, sizeof(rec_path), "%s/%s",
                  glimpse_recordings_path, device_recording_opt);
        config.recording.path = rec_path;
    }
#endif
    char *catch_err = NULL;
    data->recording_device = gm_device_open(data->log, &config, &catch_err);
    data->active_device = data->recording_device;
    if (data->recording_device) {
        configure_recording_device(data);

        char *catch_err = NULL;
        if (!gm_device_commit_config(data->recording_device, &catch_err)) {
            gm_error(data->log, "Failed to common device config: %s (falling back to opening NULL device)",
                     catch_err);
            free(catch_err);
            catch_err = NULL;

            gm_device_close(data->recording_device);
            data->recording_device = nullptr;
        }
    } else {
        gm_error(data->log, "Failed to open device: %s (falling back to opening NULL device)",
                 catch_err);
        free(catch_err);
        catch_err = NULL;
    }

    if (!data->recording_device) {
        config.type = GM_DEVICE_NULL;
        data->recording_device =
            gm_device_open(data->log, &config, NULL); // abort on error
        data->active_device = data->recording_device;

        configure_recording_device(data);

        gm_device_commit_config(data->recording_device, NULL); // abort on error
    }

    if (config.type == GM_DEVICE_TANGO ||
        config.type == GM_DEVICE_AVF)
    {
        data->realtime_ar_mode = true;
    } else {
        struct gm_ui_properties *ctx_props =
            gm_context_get_ui_properties(data->ctx);
        data->realtime_ar_mode = false;
        gm_prop_set_enum(find_prop(ctx_props, "cloud_mode"), 1);
    }

    update_ar_video_queue_len(data, 6);

    data->show_skeleton = true;
    data->show_view_cam_controls = false;

    data->target_error = 0.25f;
    data->target_progress = true;

    data->stage_stats_mode = 1; // aggregated per-frame median

    data->view_zoom = 1;

    /* FIXME: really these overrides should only affect the playback_device */
    data->dev_control_overrides.insert({
        { "frame", NULL },
        { "max_frame", NULL },
        { "loop", NULL },
        { "frame_skip", NULL },
        { "frame_throttle", NULL },
        { "<<", NULL },
        { "||>", NULL },
        { ">>", NULL },
    });

    data->initialized = true;
}

#if !defined(USE_GLFM)
static void
usage(void)
{
    printf(
"Usage glimpse_viewer [options]\n"
"\n"
"    -d,--device=DEV            Device type to use\n\n"
"                               - kinect:    Either a Kinect camera or Fakenect\n"
"                                            recording (default)\n"
"                               - recording: A glimpse_viewer recording (must\n"
"                                            pass -r/--recording option too)\n"
"                               - null:      A stub device that doesn't support\n"
"                                            any frame capture\n"
"    -r,--recording=NAME        Name or recording to play\n"
"\n"
"    -l,--log=FILE              Write logging to given file\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n"
    );

    exit(1);
}

static void
parse_args(Data *data, int argc, char **argv)
{
    int opt;

#define DEVICE_OPT              (CHAR_MAX + 1)
#define RECORDING_OPT           (CHAR_MAX + 1)

    const char *short_options="hl:d:r:";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"log",             required_argument,  0, 'l'},
        {"device",          required_argument,  0, DEVICE_OPT},
        {"recording",       required_argument,  0, RECORDING_OPT},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
            case 'h':
                usage();
                return;
            case 'l':
                log_filename_opt = strdup(optarg);
                break;
            case 'd':
                if (strcmp(optarg, "kinect") == 0)
                    device_type_opt = GM_DEVICE_KINECT;
                else if (strcmp(optarg, "recording") == 0)
                    device_type_opt = GM_DEVICE_RECORDING;
                else if (strcmp(optarg, "null") == 0)
                    device_type_opt = GM_DEVICE_NULL;
                else
                    usage();
                break;
            case 'r':
                    device_recording_opt = strdup(optarg);
                break;
            default:
                usage();
                break;
        }
    }
}
#endif

#ifdef USE_GLFM
void
glfmMain(GLFMDisplay *display)
#else  // USE_GLFW
int
main(int argc, char **argv)
#endif
{
    Data *data = new Data();
#if TARGET_OS_IOS == 1
    char *assets_root = ios_util_get_documents_path();
    char log_filename_tmp[PATH_MAX];
    snprintf(log_filename_tmp, sizeof(log_filename_tmp),
             "%s/glimpse.log", assets_root);
    data->log_fp = fopen(log_filename_tmp, "w");
    permissions_check_passed = true;
#elif defined(__ANDROID__)
    char *assets_root = strdup("/sdcard/Glimpse");
    char log_filename_tmp[PATH_MAX];
    snprintf(log_filename_tmp, sizeof(log_filename_tmp),
             "%s/glimpse.log", assets_root);
    data->log_fp = fopen(log_filename_tmp, "w");
#else
    parse_args(data, argc, argv);

    const char *assets_root_env = getenv("GLIMPSE_ASSETS_ROOT");
    char *assets_root = strdup(assets_root_env ? assets_root_env : "");

    if (log_filename_opt) {
        data->log_fp = fopen(log_filename_opt, "w");
        if (!data->log_fp) {
            fprintf(stderr, "Failed to open %s\n", log_filename_opt);
            exit(1);
        }
    } else
        data->log_fp = stderr;
#endif

    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

    gm_debug(data->log, "Glimpse Viewer");

    gm_set_assets_root(data->log, assets_root);

    if (!getenv("FAKENECT_PATH")) {
        char fakenect_path[PATH_MAX];
        snprintf(fakenect_path, sizeof(fakenect_path),
                 "%s/FakeRecording", assets_root);
        setenv("FAKENECT_PATH", fakenect_path, true);
    }

    char path_tmp[PATH_MAX];
    snprintf(path_tmp, sizeof(path_tmp),
             "%s/ViewerRecording", assets_root);
    glimpse_recordings_path = strdup(path_tmp);
    snprintf(path_tmp, sizeof(path_tmp),
             "%s/Targets", assets_root);
    glimpse_targets_path = strdup(path_tmp);

    index_recordings(data);
    index_targets(data);

    pthread_mutex_init(&data->event_queue_lock, NULL);
    pthread_cond_init(&data->event_notify_cond, NULL);
    data->events_front = new std::vector<struct event>();
    data->events_back = new std::vector<struct event>();

    reset_view(data);

#ifdef USE_GLFM
    init_winsys_glfm(data, display);
#else // USE_GLFW
    init_winsys_glfw(data);

    viewer_init(data);

    event_loop(data);

    viewer_destroy(data);

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
