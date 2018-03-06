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
#include <sys/mman.h>

#include <fcntl.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <cmath>
#include <list>
#include <forward_list>

#include <pthread.h>

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/timing.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/common/random.h>
#include <pcl/common/generate.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/angles.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/linear_least_squares_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include <epoxy/gl.h>

#define PNG_DEBUG 1
#include <png.h>
#include <setjmp.h>

#include "half.hpp"

#include "xalloc.h"
#include "wrapper_image.h"
#include "infer.h"
#include "loader.h"
#include "image_utils.h"

#include "glimpse_log.h"
#include "glimpse_mem_pool.h"
#include "glimpse_assets.h"
#include "glimpse_context.h"

#undef GM_LOG_CONTEXT
#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Tracking"
#else
#define GM_LOG_CONTEXT "ctx"
#endif
#define LOGI(...) gm_info(ctx->log, __VA_ARGS__)
#define LOGE(...) gm_error(ctx->log, __VA_ARGS__)

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)

#define joint_name(x) \
    json_object_get_string( \
        json_array_get_object(json_array(ctx->joint_map), x), "joint")

/* With this foreach macro the following block of code will have access to
 * x, y, z and off variables. (off = y * width + x)
 */
#define foreach_xy_off(width, height) \
    for (int y = 0, off = 0; y < (int)height; ++y) \
        for (int x = 0; x < (int)width; ++x, ++off)

/* A suboptimal but convenient way for us to handle image rotations inline with
 * copies/format conversions.
 *
 * x, y, width and height are coordinates and the size relative to the input
 * image.
 *
 * rot_width is the width of the rotated image
 */
#define with_rotated_rx_ry_roff(x, y, width, height, \
                                rotation, rot_width, block) \
    do { \
        int rx = x; \
        int ry = y; \
        switch (rotation) { \
        case GM_ROTATION_0: \
            rx = x; \
            ry = y; \
            break; \
        case GM_ROTATION_90: \
            rx = y; \
            ry = width - x - 1; \
            break; \
        case GM_ROTATION_180: \
            rx = width - x - 1; \
            ry = height - y - 1; \
            break; \
        case GM_ROTATION_270: \
            rx = height - y - 1; \
            ry = x; \
            break; \
        } \
        int roff = rot_width * ry + rx; \
        block \
    }while (0)


#define CLIPH(X) ((X) > 255 ? 255 : (X))
#define RGB2Y(R, G, B) ((uint8_t)CLIPH(((66 * (uint32_t)(R) + \
                                         129 * (uint32_t)(G) + \
                                         25 * (uint32_t)(B) + 128) >> 8) +  16))

using half_float::half;
using namespace pcl::common;


#define DOWNSAMPLE_1_2
//#define DOWNSAMPLE_1_4

#ifdef DOWNSAMPLE_1_4

/* One implies the other... */
#ifndef DOWNSAMPLE_1_2
#define DOWNSAMPLE_1_2
#endif

#endif

#define TRACK_FRAMES 12

enum image_format {
    IMAGE_FORMAT_X8,
    IMAGE_FORMAT_XHALF,
    IMAGE_FORMAT_XFLOAT,
};

/* XXX: fix namespace */
struct pt {
    float x, y;
};

struct color
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct color_stop
{
    float val;
    struct color color;
};

struct joint_dist
{
    float min;
    float mean;
    float max;
};

struct joint_info
{
    int n_connections;
    int *connections;
    struct joint_dist *dist;
};

struct trail_crumb
{
    char tag[32];
    int n_frames;
    void *backtrace_frame_pointers[10];
};

struct bone_info
{
    float length;
    glm::quat angle;
    int head;
    int tail;
    bool length_corrected;
    bool angle_corrected;

    bone_info() :
        length(0.f),
        head(-1),
        length_corrected(false),
        angle_corrected(false) {}
};

struct gm_skeleton {
    std::vector<struct gm_joint> joints;
    std::vector<struct bone_info> bones;
    float confidence;
    float distance;

    gm_skeleton() :
      confidence(0.f),
      distance(0.f) {}
    gm_skeleton(int n_joints) :
      joints(n_joints),
      bones(n_joints),
      confidence(0.f),
      distance(0.f) {}
};

struct gm_prediction_impl
{
    struct gm_prediction base;
    struct gm_prediction_vtable vtable;
    struct gm_mem_pool *pool;

    struct gm_context *ctx;

    uint64_t timestamp;

    struct gm_tracking_impl *tracking_history[TRACK_FRAMES];
    int n_tracking;

    struct gm_skeleton skeleton;

    pthread_mutex_t trail_lock;
    std::vector<struct trail_crumb> trail;
};

struct gm_tracking_impl
{
    struct gm_tracking base;

    struct gm_tracking_vtable vtable;

    struct gm_mem_pool *pool;

    struct gm_context *ctx;

    /* Note: these are derived from the corresponding ctx->basis_* state except
     * they take into account the device rotation at the start of tracking.
     */
    struct gm_intrinsics depth_camera_intrinsics;
    struct gm_intrinsics video_camera_intrinsics;
    struct gm_intrinsics training_camera_intrinsics;
    struct gm_extrinsics depth_to_video_extrinsics;
    bool extrinsics_set;

    struct gm_frame *frame;

    // Depth data, in meters
    float *depth;

    // Label inference data
    uint8_t *label_map;

    // Label probability tables
    float *label_probs;

    // Estimated normals for the depth buffer
    pcl::PointCloud<pcl::Normal>::Ptr normals;

    // Labels based on similar normals
    pcl::PointCloud<pcl::Label>::Ptr normal_labels;

    // Labels based on clustering after plane removal
    pcl::PointCloud<pcl::Label>::Ptr cluster_labels;

    // Whether any person clouds were tracked in this frame
    bool success;

    // Inferred joint positions
    struct gm_skeleton skeleton;

    // Legacy interface
    float *joints_processed;

    uint8_t *face_detect_buf;
    size_t face_detect_buf_width;
    size_t face_detect_buf_height;

    /* Lets us debug when we've failed to release frame resources when
     * we come to destroy our resource pools
     */
    pthread_mutex_t trail_lock;
    std::vector<struct trail_crumb> trail;
};

struct gm_context
{
    struct gm_logger *log;

    /* E.g taken during the render hook to block the context from being stopped
     * or destroyed
     */
    pthread_mutex_t liveness_lock;
    /* A pre-requisite to destroying the context is to stop tracking, which
     * will stop  the tracking thread. It's possible to only stop tracking
     * without destroying the context via gm_context_stop_tracking()
     */
    bool stopping;
    bool destroying;

    struct gm_intrinsics basis_depth_camera_intrinsics;
    struct gm_intrinsics basis_video_camera_intrinsics;
    struct gm_intrinsics basis_training_camera_intrinsics;
    struct gm_extrinsics basis_depth_to_video_extrinsics;
    bool basis_extrinsics_set;

    pthread_t detect_thread;
    dlib::frontal_face_detector detector;

    dlib::shape_predictor face_feature_detector;

    RDTree **decision_trees;
    int n_decision_trees;

    size_t grey_width;
    size_t grey_height;
    //size_t yuv_size;
    //size_t uv_buffer_offset;

    //dlib::pyramid_down<2> frame_downsampler;
    std::vector<uint8_t> grey_buffer_1_1; //original
    std::vector<uint8_t> grey_buffer_1_2; //half size

    std::vector<struct pt> landmarks;
    GLuint landmarks_program;
    GLuint attrib_landmarks_bo;
    size_t attrib_landmarks_bo_size;
    GLuint attrib_landmarks_pos;

    std::vector<uint8_t> grey_face_detect_scratch;
    std::vector<glimpse::wrapped_image<unsigned char>> grey_face_detect_wrapped_layers;

    std::atomic<bool> need_new_scaled_frame;
    pthread_mutex_t scaled_frame_cond_mutex;
    pthread_cond_t scaled_frame_available_cond;

    GLuint yuv_frame_scale_program;
    GLuint scale_program;

    enum gm_rotation current_attrib_bo_rotation_ = { GM_ROTATION_0 };
    GLuint attrib_quad_rot_scale_bo;

    GLuint attrib_quad_rot_scale_pos;
    GLuint attrib_quad_rot_scale_tex_coords;

    GLuint uniform_tex_sampler;

    GLuint cam_tex;

    bool have_portrait_downsample_fb;
    GLuint downsample_fbo;
    GLuint downsample_tex2d;


    GLuint read_back_fbo;
    GLuint read_back_pbo;

    std::vector<dlib::rectangle> last_faces;

#if 0
    int current_copy_buf;
    int current_detect_buf;
    int current_ready_buf;
#endif

    /* Throttle the face detection based on this condition variable signaled
     * when we receive a new frame.
     *
     * (In practice face detection runs much more slowly than the capture
     * rate so throttling isn't a big concern but nevertheless...)
     */

    pthread_mutex_t debug_viz_mutex;
    int grey_debug_width;
    int grey_debug_height;
    std::vector<uint8_t> grey_debug_buffer;

    /*
     * -1 means to visualize the most probable labels. Any other value
     *  says to visualize the probability of specific labels...
     */
    int debug_label;

    pthread_mutex_t skel_track_cond_mutex;
    pthread_cond_t skel_track_cond;

    /* We maintain tracking_history[] as an ordered array of tracking
     * state from [0] = newest to oldest. Initially with no tracking
     * history then n_tracking == 0 and all array entries are NULL.
     * n_tracking only increases up to TRACK_FRAMES at which point
     * tracking_history[] is a FIFO
     *
     * We only add to the history if we successfully detect a person
     *
     * We clear the history when tracking failed to detect a person
     *
     * We always store a reference to the last tracking state in
     * latest_tracking, even if we failed to detect a person.
     */
    struct gm_mem_pool *tracking_pool;
    pthread_mutex_t tracking_swap_mutex;
    struct gm_tracking_impl *tracking_history[TRACK_FRAMES];
    struct gm_tracking_impl *latest_tracking;
    int n_tracking;

    struct gm_mem_pool *prediction_pool;

    int n_labels;

    JSON_Value *joint_map;
    JIParams *joint_params;
    struct joint_info *joint_stats;
    int n_joints;

    bool depth_gap_fill;
    bool apply_depth_distortion;
    int gap_dist;
    float gap_tolerance;
    int cloud_res;
    float min_depth;
    float max_depth;
    int seg_res;
    float normal_depth_change;
    float normal_smooth;
    int min_inliers;
    float angular_threshold;
    float distance_threshold;
    float refinement_distance_threshold;
    float max_curvature;
    float cluster_tolerance;

    bool joint_refinement;
    int max_joint_predictions;
    float max_prediction_delta;
    float prediction_decay;
    float joint_move_threshold;
    float joint_max_travel;
    float joint_scale_threshold;
    bool bone_sanitisation;
    float bone_length_variance;
    float bone_rotation_variance;

    float skeleton_min_confidence;
    float skeleton_max_distance;

    int n_depth_color_stops;
    float depth_color_stops_range;
    struct color_stop *depth_color_stops;

    int n_heat_color_stops;
    float heat_color_stops_range;
    struct color_stop *heat_color_stops;

    std::vector<struct gm_ui_enumerant> label_enumerants;
    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;

    pthread_mutex_t frame_ready_mutex;
    pthread_cond_t frame_ready_cond;
    struct gm_frame *frame_ready;

    void (*event_callback)(struct gm_context *ctx,
                           struct gm_event *event,
                           void *user_data);

    void *callback_data;
};

static const char *label_names[] = {
    "head, left-side",
    "head, right-side",
    "head, top-left",
    "head, top-right",
    "neck",
    "clavicle, left",
    "clavicle, right",
    "shoulder, left",
    "upper-arm, left",
    "shoulder, right",
    "upper arm, right",
    "elbow, left",
    "forearm, left",
    "elbow, right",
    "forearm, right",
    "wrist, left",
    "hand, left",
    "wrist, right",
    "hand, right",
    "hip, left",
    "thigh, left",
    "hip, right",
    "thigh, right",
    "knee, left",
    "shin, left",
    "knee, right",
    "shin, right",
    "ankle, left",
    "toes, left",
    "ankle, right",
    "toes, right",
    "waist, left",
    "waist, right",
    "background",
};

static png_color default_palette[] = {
    { 0xff, 0x5d, 0xaa },
    { 0xd1, 0x15, 0x40 },
    { 0xda, 0x1d, 0x0e },
    { 0xdd, 0x5d, 0x1e },
    { 0x49, 0xa2, 0x24 },
    { 0x29, 0xdc, 0xe3 },
    { 0x02, 0x68, 0xc2 },
    { 0x90, 0x29, 0xf9 },
    { 0xff, 0x00, 0xcf },
    { 0xef, 0xd2, 0x37 },
    { 0x92, 0xa1, 0x3a },
    { 0x48, 0x21, 0xeb },
    { 0x2f, 0x93, 0xe5 },
    { 0x1d, 0x6b, 0x0e },
    { 0x07, 0x66, 0x4b },
    { 0xfc, 0xaa, 0x98 },
    { 0xb6, 0x85, 0x91 },
    { 0xab, 0xae, 0xf1 },
    { 0x5c, 0x62, 0xe0 },
    { 0x48, 0xf7, 0x36 },
    { 0xa3, 0x63, 0x0d },
    { 0x78, 0x1d, 0x07 },
    { 0x5e, 0x3c, 0x00 },
    { 0x9f, 0x9f, 0x60 },
    { 0x51, 0x76, 0x44 },
    { 0xd4, 0x6d, 0x46 },
    { 0xff, 0xfb, 0x7e },
    { 0xd8, 0x4b, 0x4b },
    { 0xa9, 0x02, 0x52 },
    { 0x0f, 0xc1, 0x66 },
    { 0x2b, 0x5e, 0x44 },
    { 0x00, 0x9c, 0xad },
    { 0x00, 0x40, 0xad },
    { 0x21, 0x21, 0x21 },
};


static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static char *
get_duration_ns_print_scale_suffix(uint64_t duration_ns)
{
    if (duration_ns > 1000000000)
        return (char *)"s";
    else if (duration_ns > 1000000)
        return (char *)"ms";
    else if (duration_ns > 1000)
        return (char *)"us";
    else
        return (char *)"ns";
}

static float
get_duration_ns_print_scale(uint64_t duration_ns)
{
    if (duration_ns > 1000000000)
        return duration_ns / 1e9;
    else if (duration_ns > 1000000)
        return duration_ns / 1e6;
    else if (duration_ns > 1000)
        return duration_ns / 1e3;
    else
        return duration_ns;
}

static struct color
stops_color_from_val(struct color_stop *stops,
                     int n_stops,
                     float range,
                     float val)
{
    int i = (int)((fmax(0, fmin(range, val)) / range) * n_stops) + 1;

    if (i < 1)
        i = 1;
    else if (i >= n_stops)
        i = n_stops - 1;

    float t = (val - stops[i - 1].val) / (stops[i].val - stops[i - 1].val);

    struct color col0 = stops[i - 1].color;
    struct color col1 = stops[i].color;

    float r = (1.0f - t) * col0.r + t * col1.r;
    float g = (1.0f - t) * col0.g + t * col1.g;
    float b = (1.0f - t) * col0.b + t * col1.b;

    return { (uint8_t)r, (uint8_t)g, (uint8_t)b };
}

static const uint32_t depth_rainbow[] = {
    0xffff00ff, //yellow
    0x0000ffff, //blue
    0x00ff00ff, //green
    0xff0000ff, //red
    0x00ffffff, //cyan
};

static const uint32_t heat_map_rainbow[] = {
    0x00000000, //black
    0xff0000ff, //red
    0xffff00ff, //yellow
    0x0000ffff, //blue
    0xffffffff, //white
};


static struct color
rainbow_stop_from_val(const uint32_t *rainbow, int n_rainbow_bands,
                       float val,
                       float range,
                       int n_stops) // can be > n_rainbow_bands and each cycle gets darker
{
    int stop = (int)((fmax(0, fmin(range, val)) / range) * n_stops);
    int band = ((int)stop) % n_rainbow_bands;

    uint32_t rgba = rainbow[band];

    float r = ((rgba & 0xff000000) >> 24) / 255.0f;
    float g = ((rgba & 0xff0000) >> 16) / 255.0f;
    float b = ((rgba & 0xff00) >> 8) / 255.0f;

    /* Repeated cycles through the ranbow_bands will look darker and darker... */
    float f = powf(0.8, floorf(stop / ARRAY_LEN(rainbow)));
    r *= f;
    g *= f;
    b *= f;

    return { (uint8_t)(r * 255.f), (uint8_t)(g * 255.f), (uint8_t)(b * 255.f) };
}

static void
alloc_rgb_color_stops(struct color_stop **color_stops,
                      int *n_color_stops,
                      const uint32_t *rainbow,
                      int n_rainbow_bands,
                      float range,
                      float step)
{
    int n_stops = range / step;

    *n_color_stops = n_stops;
    *color_stops = (struct color_stop *)xcalloc(sizeof(struct color_stop), n_stops);

    struct color_stop *stops = *color_stops;

    for (int i = 0; i < n_stops; i++) {
        stops[i].val = i * step;
        stops[i].color = rainbow_stop_from_val(rainbow,
                                               n_rainbow_bands,
                                               stops[i].val, range, n_stops);
    }
}

static GLuint __attribute__((unused))
load_shader(struct gm_context *ctx,
            GLenum type,
            const char *source,
            char **err)
{
    GLuint shader = glCreateShader(type);
    if (!shader)
        return 0;

    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled) {
        GLint info_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len) {
            char *buf = (char *)malloc(info_len);
            if (buf) {
                glGetShaderInfoLog(shader, info_len, NULL, buf);
                gm_throw(ctx->log, err, "Could not compile shader %d:\n%s\n",
                         type, buf);
                free(buf);
            }
            glDeleteShader(shader);
            shader = 0;
        }
    }

    return shader;
}

static GLuint __attribute__((unused))
create_program(struct gm_context *ctx,
               const char *vertex_source,
               const char *fragment_source,
               char **err)
{
    GLuint vertex_shader = load_shader(ctx, GL_VERTEX_SHADER, vertex_source, err);
    if (!vertex_shader)
        return 0;

    GLuint fragment_shader = load_shader(ctx, GL_FRAGMENT_SHADER, fragment_source, err);
    if (!fragment_shader) {
        glDeleteShader(vertex_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    if (!program) { 
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
        GLint buf_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &buf_length);
        if (buf_length) {
            char *buf = (char *)malloc(buf_length);
            if (buf) {
                glGetProgramInfoLog(program, buf_length, NULL, buf);
                gm_throw(ctx->log, err, "Could not link program:\n%s\n", buf);
                free(buf);
            }
        }
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void
gm_context_detect_faces(struct gm_context *ctx, struct gm_tracking_impl *tracking)
{
    if (!tracking->face_detect_buf) {
        LOGI("NULL tracking->face_detect_buf");
        return;
    }

    uint64_t start, end, duration_ns;
    std::vector<dlib::rectangle> face_rects(0);
    glimpse::wrapped_image<unsigned char> grey_img;
    dlib::rectangle buf_rect(tracking->face_detect_buf_width, tracking->face_detect_buf_height);

    LOGI("New camera frame to process");

    if (ctx->last_faces.size()) {

        LOGI("Searching %d region[s] for faces", (int)ctx->last_faces.size());

        for (dlib::rectangle &rect : ctx->last_faces) {

            rect = dlib::grow_rect(static_cast<dlib::rectangle&>(rect), (long)((float)rect.width() * 0.4f));
            rect.intersect(buf_rect);

            grey_img.wrap(rect.width(),
                          rect.height(),
                          tracking->face_detect_buf_width, //stride
                          static_cast<unsigned char *>(tracking->face_detect_buf +
                                                       rect.top() * tracking->face_detect_buf_width +
                                                       rect.left()));
            LOGI("Starting constrained face detection with %dx%d sub image",
                 (int)rect.width(), (int)rect.height());
            start = get_time();
            std::vector<dlib::rectangle> dets = ctx->detector(grey_img);
            end = get_time();
            duration_ns = end - start;
            LOGI("Number of detected faces = %d, %.3f%s",
                 (int)dets.size(),
                 get_duration_ns_print_scale(duration_ns),
                 get_duration_ns_print_scale_suffix(duration_ns));

            if (dets.size() != 1) {
                LOGE("Constrained search was expected to find exactly one face - fallback");
                face_rects.resize(0);
                break;
            }

            dlib::rectangle mapped_rect =
                dlib::translate_rect(dets[0], rect.left(), rect.top());

            face_rects.push_back(mapped_rect);
        }
    }

    /* Even if not used for full frame face detection, we still want
     * an image for the full frame for the landmark detection...
     */
    grey_img.wrap(tracking->face_detect_buf_width,
                  tracking->face_detect_buf_height,
                  tracking->face_detect_buf_width, //stride
                  static_cast<unsigned char *>(tracking->face_detect_buf));

    /* Fall back to checking full frame if the number of detected
     * faces has changed
     */
    if (face_rects.size() != ctx->last_faces.size() ||
        face_rects.size() == 0)
    {
        LOGI("Starting face detection with %dx%d image",
             (int)tracking->face_detect_buf_width, (int)tracking->face_detect_buf_height);
        start = get_time();
        face_rects = ctx->detector(grey_img);
        end = get_time();
        duration_ns = end - start;
        LOGI("Number of detected faces = %d, %.3f%s",
             (int)face_rects.size(),
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));
    }

    ctx->last_faces = face_rects;


    std::vector<struct pt> landmarks(0);

    for (unsigned i = 0; i < ctx->last_faces.size(); i++) {
        struct pt point;
        dlib::rectangle rect;

        start = get_time();
        dlib::full_object_detection features = ctx->face_feature_detector(grey_img, ctx->last_faces[i]);
        end = get_time();
        duration_ns = end - start;

        LOGI("Detected %d face %d features in %.3f%s",
             (int)features.num_parts(),
             (int)i,
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));

        /*
         * Bounding box
         */
        point.x = features.get_rect().left();
        point.y = features.get_rect().bottom();
        landmarks.push_back(point);
        point.x = features.get_rect().left();
        point.y = features.get_rect().top();
        landmarks.push_back(point);
        landmarks.push_back(point);
        point.x = features.get_rect().right();
        point.y = features.get_rect().top();
        landmarks.push_back(point);
        landmarks.push_back(point);
        point.x = features.get_rect().right();
        point.y = features.get_rect().bottom();
        landmarks.push_back(point);
        landmarks.push_back(point);
        point.x = features.get_rect().left();
        point.y = features.get_rect().bottom();
        landmarks.push_back(point);

        /*
         * Chin line
         */

        point.x = features.part(0).x();
        point.y = features.part(0).y();
        landmarks.push_back(point);

        for (int j = 1; j < 16; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(16).x();
        point.y = features.part(16).y();
        landmarks.push_back(point);

        /*
         * Left eyebrow
         */

        point.x = features.part(17).x();
        point.y = features.part(17).y();
        landmarks.push_back(point);

        for (int j = 18; j < 21; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(21).x();
        point.y = features.part(21).y();
        landmarks.push_back(point);

        /*
         * Right eyebrow
         */

        point.x = features.part(22).x();
        point.y = features.part(22).y();
        landmarks.push_back(point);

        for (int j = 23; j < 26; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(26).x();
        point.y = features.part(26).y();
        landmarks.push_back(point);

        /*
         * Nose
         */

        point.x = features.part(27).x();
        point.y = features.part(27).y();
        landmarks.push_back(point);

        for (int j = 28; j < 35; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(35).x();
        point.y = features.part(35).y();
        landmarks.push_back(point);

        /*
         * Left eye
         */

        point.x = features.part(36).x();
        point.y = features.part(36).y();
        landmarks.push_back(point);

        for (int j = 37; j < 42; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(36).x();
        point.y = features.part(36).y();
        landmarks.push_back(point);

        /*
         * Right eye
         */

        point.x = features.part(42).x();
        point.y = features.part(42).y();
        landmarks.push_back(point);

        for (int j = 43; j < 48; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(42).x();
        point.y = features.part(42).y();
        landmarks.push_back(point);

        /*
         * Mouth (outer)
         */

        point.x = features.part(48).x();
        point.y = features.part(48).y();
        landmarks.push_back(point);

        for (int j = 49; j < 60; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(48).x();
        point.y = features.part(48).y();
        landmarks.push_back(point);

        /*
         * Mouth (inner)
         */

        point.x = features.part(60).x();
        point.y = features.part(60).y();
        landmarks.push_back(point);

        for (int j = 61; j < 68; j++) {
            point.x = features.part(j).x();
            point.y = features.part(j).y();
            landmarks.push_back(point);
            landmarks.push_back(point);
        }

        point.x = features.part(60).x();
        point.y = features.part(60).y();
        landmarks.push_back(point);

    }

    /* Convert into normalized device coordinates */
    for (unsigned i = 0; i < landmarks.size(); i++) {
        landmarks[i].x = (landmarks[i].x / (float)tracking->face_detect_buf_width) * 2.f - 1.f;
        landmarks[i].y = (landmarks[i].y / (float)tracking->face_detect_buf_height) * -2.f + 1.f;
    }

    /* XXX: This mutex is reused for the grey debug buffer and the
     * ctx->landmarks array
     */
    pthread_mutex_lock(&ctx->debug_viz_mutex);
    ctx->landmarks.swap(landmarks);
    pthread_mutex_unlock(&ctx->debug_viz_mutex);

#ifdef VISUALIZE_DETECT_FRAME
    {
        uint64_t start = get_time();
        pthread_mutex_lock(&ctx->debug_viz_mutex);
        /* Save the frame to display for debug too... */
        grey_debug_buffer_.resize(tracking->face_detect_buf_width * tracking->face_detect_buf_height);
        memcpy(&grey_debug_buffer_[0], tracking->face_detect_buf, grey_debug_buffer_.size());
        grey_debug_width_ = tracking->face_detect_buf_width;
        grey_debug_height_ = tracking->face_detect_buf_height;
        pthread_mutex_unlock(&ctx->debug_viz_mutex);
        uint64_t end = get_time();
        uint64_t duration_ns = end - start;
        LOGE("Copied face detect buffer for debug overlay in %.3f%s",
             get_duration_ns_print_scale(duration_ns),
             get_duration_ns_print_scale_suffix(duration_ns));
    }
#endif
}

static inline float
distance_between(float *point1, float *point2)
{
    return sqrtf(powf(point1[0] - point2[0], 2.f) +
                 powf(point1[1] - point2[1], 2.f) +
                 powf(point1[2] - point2[2], 2.f));
}

static bool
compare_skeletons(const struct gm_skeleton &skel1,
                  const struct gm_skeleton &skel2)
{
    return (skel1.distance < skel2.distance) ||
           (skel1.distance == skel2.distance &&
            skel1.confidence > skel2.confidence);
}

static void
build_bones(struct gm_context *ctx,
            struct gm_skeleton &skeleton,
            bool reset = true,
            int joint_no = 0,
            int last_joint_no = -1)
{
    if (reset) {
        skeleton.bones[joint_no].head = -1;
        skeleton.bones[joint_no].length_corrected = false;
        skeleton.bones[joint_no].angle_corrected = false;
    }

    if (joint_no != last_joint_no) {
        if (skeleton.joints[joint_no].confidence > 0.f) {
            struct gm_joint &tail = skeleton.joints[joint_no];
            if (last_joint_no != -1 &&
                skeleton.joints[last_joint_no].confidence > 0) {
                struct gm_joint &head = skeleton.joints[last_joint_no];
                float dist = distance_between(&tail.x, &head.x);

                skeleton.bones[joint_no].length = dist;
                skeleton.bones[joint_no].head = last_joint_no;
                skeleton.bones[joint_no].tail = joint_no;

                if (skeleton.bones[last_joint_no].head >= 0) {
                    struct gm_joint &last_head =
                        skeleton.joints[skeleton.bones[last_joint_no].head];

                    // Calculate the angle between this bone and its parent
                    glm::vec3 bone_vec = glm::normalize(
                        glm::vec3(tail.x - head.x,
                                  tail.y - head.y,
                                  tail.z - head.z));
                    glm::vec3 parent_vec = glm::normalize(
                        glm::vec3(head.x - last_head.x,
                                  head.y - last_head.y,
                                  head.z - last_head.z));

                    glm::vec3 axis = glm::normalize(glm::cross(bone_vec,
                                                               parent_vec));
                    float angle = acosf(glm::dot(bone_vec, parent_vec));

                    skeleton.bones[joint_no].angle =
                        glm::angleAxis(angle, axis);

#if 0
                    {
                        // Debugging
                        glm::vec3 tb = (parent_vec *
                            glm::mat3_cast(skeleton.bones[joint_no].angle)) *
                            dist;
                        gm_debug(ctx->log, "XXX tail: %.2f, %.2f, %.2f, "
                                 "transformed parent: %.2f, %.2f, %.2f",
                                 tail.x, tail.y, tail.z,
                                 tb.x + head.x, tb.y + head.y, tb.z + head.z);
                    }
#endif
                }
            }
        }
    }

    for (int i = 0; i < ctx->joint_stats[joint_no].n_connections; ++i) {
        if (ctx->joint_stats[joint_no].connections[i] == last_joint_no) {
            continue;
        }
        build_bones(ctx, skeleton, reset,
                    ctx->joint_stats[joint_no].connections[i], joint_no);
    }
}

static void
build_skeleton(struct gm_context *ctx,
               InferredJoints *result,
               struct gm_skeleton &skeleton,
               int joint_no = 0,
               int last_joint_no = -1)
{
    // Add the highest confidence joint to the skeleton and track the sum
    // confidence and deviation from the expected joint distances.
    if (joint_no != last_joint_no) {
        if (result->joints[joint_no]) {
            Joint *tail = (Joint *)result->joints[joint_no]->data;

            if (last_joint_no != -1 &&
                skeleton.joints[last_joint_no].confidence > 0) {
                // Get the distance info for the bone between joint_no and
                // last_joint_no and see if this bone falls outside of the
                // expected distance between joints.
                struct gm_joint &head = skeleton.joints[last_joint_no];
                const struct joint_dist &joint_dist =
                    ctx->joint_stats[last_joint_no].dist[joint_no];

                float dist = distance_between(&tail->x, &head.x);

                if (dist < joint_dist.min) {
                    skeleton.distance += powf(joint_dist.min - dist, 2.f);
                } else if (dist > joint_dist.max) {
                    skeleton.distance += powf(dist - joint_dist.max, 2.f);
                }
            }

            skeleton.confidence += tail->confidence;
            skeleton.joints[joint_no].x = tail->x;
            skeleton.joints[joint_no].y = tail->y;
            skeleton.joints[joint_no].z = tail->z;
            skeleton.joints[joint_no].confidence = tail->confidence;
            skeleton.joints[joint_no].predicted = false;
        } else {
            // We don't want skeletons with missing joints to rank higher than
            // full skeletons, so add some large amount to the distance
            skeleton.distance += 10.f;
            skeleton.joints[joint_no].confidence = 0;
        }
    }

    // Follow the connections for this joint to build up a whole skeleton
    for (int i = 0; i < ctx->joint_stats[joint_no].n_connections; ++i) {
        if (ctx->joint_stats[joint_no].connections[i] == last_joint_no) {
            continue;
        }
        build_skeleton(ctx, result, skeleton,
                       ctx->joint_stats[joint_no].connections[i], joint_no);
    }
}

static void
refine_skeleton(struct gm_context *ctx,
                InferredJoints *result,
                struct gm_skeleton &skeleton)
{
    if (!ctx->joint_stats || !ctx->joint_refinement) {
        return;
    }

    gm_debug(ctx->log, "Prime skeleton confidence: %f, distance: %f",
             skeleton.confidence, skeleton.distance);

    // For each joint, we look at the score of the skeleton using each
    // joint cluster and if it scores higher than the most confident
    // joint, we replace that joint and continue.
    bool is_refined = false;
    for (int j = 0; j < ctx->n_joints; ++j) {
        if (!result->joints[j] || !result->joints[j]->next) {
            continue;
        }

        for (LList *l = result->joints[j]->next; l; l = l->next) {
            struct gm_skeleton candidate_skeleton(result->n_joints);
            Joint *joint = (Joint *)l->data;
            candidate_skeleton.joints[j].x = joint->x;
            candidate_skeleton.joints[j].y = joint->y;
            candidate_skeleton.joints[j].z = joint->z;
            candidate_skeleton.joints[j].confidence = joint->confidence;
            candidate_skeleton.joints[j].predicted = false;
            candidate_skeleton.confidence += joint->confidence;
            build_skeleton(ctx, result, candidate_skeleton, j, j);

            if (compare_skeletons(candidate_skeleton, skeleton)) {
                std::swap(skeleton, candidate_skeleton);
                is_refined = true;
            }
        }
    }

    if (is_refined) {
        gm_debug(ctx->log, "Refined skeleton confidence: %f, distance: %f",
                 skeleton.confidence, skeleton.distance);
    }
}

static uint64_t
calculate_decayed_timestamp(uint64_t base, uint64_t timestamp,
                            float max_delta, float delta_decay)
{
    uint64_t max_delta_64 = (uint64_t)(max_delta * 1000000.0);
    uint64_t delta = (timestamp < base) ?
        base - timestamp : timestamp - base;
    if (delta <= max_delta_64) {
        return timestamp;
    }

    uint64_t max_decay = (uint64_t)(max_delta *
                                    delta_decay * 1000000.0);
    uint64_t decay_time = std::min(delta - max_delta_64, max_decay);
    uint64_t decay = (uint64_t)(sqrt(decay_time / (double)max_decay) *
                                max_decay);

    return (timestamp < base) ?
        (base - max_delta_64) - decay :
        base + max_delta_64 + decay;
}

static void
interpolate_joints(struct gm_joint &a, struct gm_joint &b, float t,
                   struct gm_joint &out)
{
    out.x = a.x + (b.x - a.x) * t;
    out.y = a.y + (b.y - a.y) * t;
    out.z = a.z + (b.z - a.z) * t;
}

static void
sanitise_skeleton(struct gm_context *ctx,
                  struct gm_skeleton &skeleton,
                  uint64_t timestamp,
                  int parent_head = 0)
{
    if (!ctx->n_tracking) {
        return;
    }

    // Cap the time distance used when making time-based extrapolations.
    timestamp = calculate_decayed_timestamp(
        ctx->tracking_history[0]->frame->timestamp,
        timestamp, ctx->max_prediction_delta, ctx->prediction_decay);

    // We process the skeleton with regards to the 'parent' bone. At the
    // parent, we just have absolute position to look at, so we make sure it
    // hasn't moved too far too quickly and use linear interpolation
    // (TODO: with dampening) if it has.
    //
    // TODO: This first section just linearly interpolates between old positions
    //       with no regard to bone length or previous transformations. Length
    //       will be corrected in the next loop that decides on bone angles,
    //       but we could still be cleverer than this.
    for (int i = 0; i <= ctx->joint_stats[parent_head].n_connections; ++i) {
        int joint = i ?
            ctx->joint_stats[parent_head].connections[i-1] : parent_head;
        struct gm_joint &parent_joint = skeleton.joints[joint];

        // Find the index of the last 2 unpredicted parent joint positions
        int np = 0;
        int prev_frames = 0;
        struct gm_tracking_impl *prev[2];
        for (int i = 0; i < ctx->n_tracking &&
             np < ctx->max_joint_predictions; ++i) {
            if (ctx->tracking_history[i]->skeleton.
                joints[joint].predicted) {
                ++np;
                continue;
            }

            if (prev_frames < 2) {
                prev[prev_frames++] = ctx->tracking_history[i];
            }
        }

        if (prev_frames == 2 && np < ctx->max_joint_predictions) {
            struct gm_joint &prev0 = prev[0]->skeleton.joints[joint];
            struct gm_joint &prev1 = prev[1]->skeleton.joints[joint];

            float time = (float)((timestamp - prev[0]->frame->timestamp) / 1e9);
            float distance = distance_between(&parent_joint.x, &prev0.x) / time;

            float prev_time = (float)
                ((prev[0]->frame->timestamp -
                  prev[1]->frame->timestamp) / 1e9);
            float prev_dist = distance_between(&prev0.x, &prev1.x) / prev_time;

            if (distance > ctx->joint_move_threshold &&
                (distance > ctx->joint_max_travel ||
                 distance > (prev_dist * ctx->joint_scale_threshold))) {
                glm::vec3 delta = glm::vec3(prev0.x, prev0.y, prev0.z) -
                                  glm::vec3(prev1.x, prev1.y, prev1.z);
                glm::vec3 prediction =
                    glm::vec3(prev1.x, prev1.y, prev1.z) +
                    (delta / prev_time) * time;

                float new_distance = distance_between(&prediction.x, &prev0.x);

                // If linear interpolation from the last joint confidence
                // joint positions is closer to the previous position than the
                // current tracking position, replace it.
                if (new_distance < distance) {
                    LOGI("Parent joint %s replaced (%.2f, %.2f, %.2f)->"
                         "(%.2f, %.2f, %.2f) (prev: %.2f, %.2f, %.2f) "
                         "(%.2f, %.2fx, c %.2f)",
                         joint_name(joint),
                         parent_joint.x, parent_joint.y, parent_joint.z,
                         prediction.x, prediction.y, prediction.z,
                         prev0.x, prev0.y, prev0.z,
                         distance, distance / prev_dist, parent_joint.confidence);
                    parent_joint.x = prediction.x;
                    parent_joint.y = prediction.y;
                    parent_joint.z = prediction.z;
                    parent_joint.predicted = true;
                }
            }
        }
    }

    // Calculate the bone metadata
    build_bones(ctx, skeleton);

    // For each bone, we compare the bone length and look at the change in
    // angular acceleration. Bone length shouldn't change of course, but
    // tracking isn't perfect, so we allow some squishiness. If either exceed
    // the set thresholds, we replace this bone with a prediction based on
    // previous confident bones.
    for (std::vector<struct bone_info>::iterator it = skeleton.bones.begin();
         it != skeleton.bones.end(); ++it) {
        struct bone_info &bone = *it;

        if (bone.head < 0) {
            continue;
        }

        LOGI("Bone (%s->%s) (%.2f, %.2f, %.2f)->(%.2f,%.2f,%.2f) l: %.2f",
             joint_name(bone.head), joint_name(bone.tail),
             skeleton.joints[bone.head].x,
             skeleton.joints[bone.head].y,
             skeleton.joints[bone.head].z,
             skeleton.joints[bone.tail].x,
             skeleton.joints[bone.tail].y,
             skeleton.joints[bone.tail].z,
             bone.length);

        // Find the last bones without corrections.
        // TODO: We probably want to set limits on how many corrections are
        //       reasonable to make within a particular tracking history, or
        //       contiguously.
        struct gm_tracking_impl *prev_length = NULL;
        struct gm_tracking_impl *prev_angle = NULL;
        for (int i = 0; i < ctx->n_tracking; ++i) {
            struct bone_info &prev_bone =
                ctx->tracking_history[i]->skeleton.bones[bone.tail];
            if (prev_bone.head < 0) {
                continue;
            }
            if (!prev_bone.length_corrected && !prev_length) {
                prev_length = ctx->tracking_history[i];
            }
            if (!prev_bone.angle_corrected && !prev_angle) {
                prev_angle = ctx->tracking_history[i];
            }
        }

        if (prev_length) {
            struct bone_info &prev_bone =
                prev_length->skeleton.bones[bone.tail];

            // If the bone length has changed significantly, adjust the length
            // so that it remains within our acceptable bounds of variation.
            float bone_length_diff = fabsf(bone.length - prev_bone.length);
            if (bone_length_diff > ctx->bone_length_variance) {
                float new_length = (bone.length > prev_bone.length) ?
                    prev_bone.length + ctx->bone_length_variance :
                    prev_bone.length - ctx->bone_length_variance;

                glm::vec3 new_tail =
                    glm::vec3(skeleton.joints[bone.head].x,
                              skeleton.joints[bone.head].y,
                              skeleton.joints[bone.head].z) +
                    (glm::normalize(
                         glm::vec3(skeleton.joints[bone.tail].x -
                                   skeleton.joints[bone.head].x,
                                   skeleton.joints[bone.tail].y -
                                   skeleton.joints[bone.head].y,
                                   skeleton.joints[bone.tail].z -
                                   skeleton.joints[bone.head].z)) * new_length);

                skeleton.joints[bone.tail].x = new_tail.x;
                skeleton.joints[bone.tail].y = new_tail.y;
                skeleton.joints[bone.tail].z = new_tail.z;
                skeleton.joints[bone.tail].predicted = true;
                bone.length_corrected = true;

                // Rebuild bone info now the joint has changed
                build_bones(ctx, skeleton, false);

                gm_debug(ctx->log, "Bone length correction (%s->%s) "
                         "(%.2f, %.2f, %.2f)->(%.2f, %.2f, %.2f) l: %.2f",
                         joint_name(bone.head), joint_name(bone.tail),
                         skeleton.joints[bone.head].x,
                         skeleton.joints[bone.head].y,
                         skeleton.joints[bone.head].z,
                         skeleton.joints[bone.tail].x,
                         skeleton.joints[bone.tail].y,
                         skeleton.joints[bone.tail].z,
                         bone.length);
            }
        }

        // If this bone has no parent bone, we can't correct its angle
        int parent_head = skeleton.bones[bone.head].head;
        if (parent_head < 0) {
            continue;
        }

        if (prev_angle) {
            struct gm_skeleton &prev_skel = prev_angle->skeleton;

            // Calculate the angle between the bone and the last uncorrected
            // bone.
            glm::vec3 bone_vec = glm::vec3(skeleton.joints[bone.tail].x -
                                           skeleton.joints[bone.head].x,
                                           skeleton.joints[bone.tail].y -
                                           skeleton.joints[bone.head].y,
                                           skeleton.joints[bone.tail].z -
                                           skeleton.joints[bone.head].z);
            glm::vec3 prev_vec = glm::vec3(prev_skel.joints[bone.tail].x -
                                           prev_skel.joints[bone.head].x,
                                           prev_skel.joints[bone.tail].y -
                                           prev_skel.joints[bone.head].y,
                                           prev_skel.joints[bone.tail].z -
                                           prev_skel.joints[bone.head].z);

            float angle = glm::degrees(acosf(
                glm::dot(glm::normalize(bone_vec), glm::normalize(prev_vec))));
            while (angle > 180.f) angle -= 360.f;
            float time = (float)
                ((timestamp - prev_angle->frame->timestamp) / 1e9);

            // If the bone angle has changed quicker than we expected, use the
            // previous bone angle.
            float angle_delta = fabsf(angle) / time;
            if (angle_delta > ctx->bone_rotation_variance) {
#if 0
                // It feels like mixing in some of the angle from the spurious
                // tracking might be nice, but generally, if we're doing this
                // correction it was because the tracked point is nonsense.
                glm::mat3 rotate = glm::mat3_cast(
                    glm::slerp(prev_bone.angle, bone.angle,
                               (ctx->bone_rotation_variance / angle_delta)));
#else
                // We don't care about using a corrected bone for this, we want
                // the angle to change smoothly regardless of whether it's
                // based on a corrected value or not.
                struct bone_info &abs_prev_bone =
                    ctx->tracking_history[0]->skeleton.bones[bone.tail];
                glm::mat3 rotate = glm::mat3_cast(abs_prev_bone.angle);
#endif

                struct bone_info &parent_bone = skeleton.bones[bone.head];
                glm::vec3 parent_vec = glm::normalize(
                    glm::vec3(skeleton.joints[parent_bone.tail].x -
                              skeleton.joints[parent_bone.head].x,
                              skeleton.joints[parent_bone.tail].y -
                              skeleton.joints[parent_bone.head].y,
                              skeleton.joints[parent_bone.tail].z -
                              skeleton.joints[parent_bone.head].z));

                glm::vec3 new_tail = ((parent_vec * rotate) * bone.length);
                new_tail.x += skeleton.joints[bone.head].x;
                new_tail.y += skeleton.joints[bone.head].y;
                new_tail.z += skeleton.joints[bone.head].z;

                skeleton.joints[bone.tail].x = new_tail.x;
                skeleton.joints[bone.tail].y = new_tail.y;
                skeleton.joints[bone.tail].z = new_tail.z;
                skeleton.joints[bone.tail].predicted = true;
                bone.angle_corrected = true;

                // Rebuild bone info now the joint has changed
                build_bones(ctx, skeleton, false);

                gm_debug(ctx->log, "Bone angle correction (%s->%s) "
                         "(%.2f, %.2f, %.2f)->(%.2f, %.2f, %.2f) "
                         "l: %.2f, a: %.2f",
                         joint_name(bone.head), joint_name(bone.tail),
                         skeleton.joints[bone.head].x,
                         skeleton.joints[bone.head].y,
                         skeleton.joints[bone.head].z,
                         skeleton.joints[bone.tail].x,
                         skeleton.joints[bone.tail].y,
                         skeleton.joints[bone.tail].z,
                         bone.length, angle_delta);
            }
        }
    }
}

template<typename PointT>
class PlaneComparator: public pcl::Comparator<PointT>
{
  public:
    typedef typename pcl::Comparator<PointT>::PointCloud PointCloud;
    typedef typename pcl::Comparator<PointT>::PointCloudConstPtr
        PointCloudConstPtr;

    typedef boost::shared_ptr<PlaneComparator<PointT>> Ptr;
    typedef boost::shared_ptr<const PlaneComparator<PointT>> ConstPtr;

    using pcl::Comparator<PointT>::input_;

    PlaneComparator()
      : coeffs_(0.f, 1.f, 0.f, 1.f),
        distance_threshold_(0.03f) {
    }

    virtual
    ~PlaneComparator() {
    }

    inline void
    setPlaneCoefficients(Eigen::Vector4f &coeffs) {
        coeffs_ = coeffs;
    }

    inline Eigen::Vector4f &
    getPlaneCoefficients() {
        return coeffs_;
    }

    inline void
    setDistanceThreshold(float distance_threshold) {
        distance_threshold_ = distance_threshold;
    }

    inline float
    getDistanceThreshold() const {
        return distance_threshold_;
    }

    virtual bool
    compare (int idx1, int idx2) const {
        Eigen::Vector4f e_pt(input_->points[idx1].x,
                             input_->points[idx1].y,
                             input_->points[idx1].z, 1.f);
        return coeffs_.dot(e_pt) < distance_threshold_;
    }

  protected:
    Eigen::Vector4f coeffs_;
    float distance_threshold_;
};

template<typename PointT, typename PointNT>
class DepthComparator: public pcl::Comparator<PointT>
{
  public:
    typedef typename pcl::Comparator<PointT>::PointCloud PointCloud;
    typedef typename pcl::Comparator<PointT>::PointCloudConstPtr
        PointCloudConstPtr;

    typedef typename pcl::PointCloud<PointNT> PointCloudN;
    typedef typename PointCloudN::Ptr PointCloudNPtr;
    typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

    typedef boost::shared_ptr<DepthComparator<PointT, PointNT>> Ptr;
    typedef boost::shared_ptr<const DepthComparator<PointT, PointNT>> ConstPtr;

    using pcl::Comparator<PointT>::input_;

    DepthComparator()
      : normals_()
      , depth_threshold_(0.03f) {
    }

    virtual
    ~DepthComparator() {
    }

    inline void
    setInputNormals(const PointCloudNConstPtr &normals) {
        normals_ = normals;
    }

    inline PointCloudNConstPtr
    getInputNormals () const {
        return normals_;
    }

    inline void
    setDepthThreshold(float depth_threshold) {
        depth_threshold_ = depth_threshold;
    }

    inline float
    getDepthThreshold() const {
        return depth_threshold_;
    }

    virtual bool
    compare (int idx1, int idx2) const {
        return fabsf(input_->points[idx1].z - input_->points[idx2].z) <
            depth_threshold_;
    }

  protected:
    PointCloudNConstPtr normals_;
    float depth_threshold_;
};

static bool
gm_context_track_skeleton(struct gm_context *ctx,
                          struct gm_tracking_impl *tracking)
{
    uint64_t start, end, duration;

    float nan = std::numeric_limits<float>::quiet_NaN();
    pcl::PointXYZ invalid_pt(nan, nan, nan);

    // X increases to the right
    // Y increases downwards
    // Z increases outwards

    // Project depth buffer into cloud and filter out points that are too
    // near/far.
    start = get_time();
    pcl::PointCloud<pcl::PointXYZ>::Ptr hires_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    hires_cloud->width = tracking->depth_camera_intrinsics.width / ctx->cloud_res;
    hires_cloud->height = tracking->depth_camera_intrinsics.height / ctx->cloud_res;
    hires_cloud->points.resize(hires_cloud->width * hires_cloud->height);
    hires_cloud->is_dense = false;

    int n_points = 0;

    float inv_fx = 1.0f / tracking->depth_camera_intrinsics.fx;
    float inv_fy = -1.0f / tracking->depth_camera_intrinsics.fy;
    float cx = tracking->depth_camera_intrinsics.cx;
    float cy = tracking->depth_camera_intrinsics.cy;

    foreach_xy_off(hires_cloud->width, hires_cloud->height) {
        int doff = (y * ctx->cloud_res) * tracking->depth_camera_intrinsics.width +
                   (x * ctx->cloud_res);
        float depth = tracking->depth[doff];
        if (std::isnormal(depth) &&
            depth >= ctx->min_depth &&
            depth < ctx->max_depth) {
            float dx = ((x * ctx->cloud_res) - cx) * depth * inv_fx;
            float dy = -((y * ctx->cloud_res) - cy) * depth * inv_fy;
            hires_cloud->points[off].x = dx;
            hires_cloud->points[off].y = dy;
            hires_cloud->points[off].z = depth;
            ++n_points;
        } else {
            hires_cloud->points[off] = invalid_pt;
        }
    }

    // Person detection can happen in a sparser cloud made from a downscaled
    // version of the depth buffer. This is significantly cheaper than using a
    // voxel grid, which would produce better results but take a lot longer
    // doing so and give us less useful data structures.
    int n_lores_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr lores_cloud;
    if (ctx->seg_res > 1) {
        lores_cloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>(
              new pcl::PointCloud<pcl::PointXYZ>);
        lores_cloud->width = hires_cloud->width / ctx->seg_res;
        lores_cloud->height = hires_cloud->height / ctx->seg_res;
        lores_cloud->points.resize(lores_cloud->width * lores_cloud->height);
        lores_cloud->is_dense = false;

        n_lores_points = 0;
        foreach_xy_off(lores_cloud->width, lores_cloud->height) {
            int hoff = (y * ctx->seg_res) * hires_cloud->width + (x * ctx->seg_res);
            lores_cloud->points[off] = hires_cloud->points[hoff];
            if (!std::isnan(lores_cloud->points[off].z)) {
                ++n_lores_points;
            }
        }
    } else {
        lores_cloud = hires_cloud;
        n_lores_points = n_points;
    }

    end = get_time();
    duration = end - start;
    LOGI("Projection (%d points, %d low-res) took (%.3f%s)\n",
         n_points, n_lores_points,
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Remove dense planes above a certain size
    start = get_time();

    // Estimate normals of depth cloud
    tracking->normals = pcl::PointCloud<pcl::Normal>::Ptr(
        new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
#if 0
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    // COVARIANCE_MATRIX, AVERAGE_3D_GRADIENT, AVERAGE_DEPTH_CHANGE
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(ctx->normal_depth_change);
    ne.setNormalSmoothingSize(ctx->normal_smooth * 10);
    //ne.setRectSize(lores_cloud->width, lores_cloud->height);
    ne.setInputCloud(lores_cloud);
    ne.compute(*tracking->normals);
#elif 1
    pcl::LinearLeastSquaresNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNormalSmoothingSize(ctx->normal_smooth);
    ne.setMaxDepthChangeFactor(ctx->normal_depth_change);
    ne.setKSearch(0);
    ne.setRadiusSearch(ctx->normal_depth_change);
    ne.setDepthDependentSmoothing(false);
    ne.setInputCloud(lores_cloud);
    ne.compute(*tracking->normals);
#else
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(lores_cloud);

#if 0
    pcl::search::KdTree<pcl::PointXYZ>::Ptr
      tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setSortedResults(false);
    tree->setInputCloud(lores_cloud);
    ne.setSearchMethod(tree);
#endif

    ne.setRadiusSearch(ctx->normal_depth_change);
    ne.compute(*tracking->normals);
#endif

    end = get_time();
    duration = end - start;
    LOGI("Normal estimation took (%.3f%s)\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Segment clouds into planes
    start = get_time();

    pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label>
        seg;
    seg.setMinInliers((unsigned)ctx->min_inliers);
    seg.setAngularThreshold(pcl::deg2rad(ctx->angular_threshold));
    seg.setDistanceThreshold(ctx->distance_threshold);
    seg.setMaximumCurvature(ctx->max_curvature);
    seg.setInputCloud(lores_cloud);
    seg.setInputNormals(tracking->normals);

#if 1
    pcl::EuclideanPlaneCoefficientComparator<pcl::PointXYZ, pcl::Normal>::Ptr
        plane_comparator(
            new pcl::EuclideanPlaneCoefficientComparator<pcl::PointXYZ,
                                                         pcl::Normal>);
    seg.setComparator(plane_comparator);
#endif

#if 1
    pcl::PlaneRefinementComparator<pcl::PointXYZ, pcl::Normal, pcl::Label>::Ptr
        refinement_comparator(
            new pcl::PlaneRefinementComparator<pcl::PointXYZ, pcl::Normal,
                                               pcl::Label>);
    seg.setRefinementComparator(refinement_comparator);
    refinement_comparator->
        setDistanceThreshold(ctx->refinement_distance_threshold, true);
#endif

    std::vector<pcl::ModelCoefficients> plane_coeffs;
    std::vector<pcl::PointIndices> plane_indices;
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>
        plane_centroids;
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>
        plane_covariances;
    tracking->normal_labels =
        pcl::PointCloud<pcl::Label>::Ptr(new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> plane_label_indices;
    seg.segment(plane_coeffs, plane_indices, plane_centroids,
                plane_covariances, *tracking->normal_labels, plane_label_indices);
    seg.refine(plane_coeffs, plane_indices, plane_centroids,
               plane_covariances, tracking->normal_labels, plane_label_indices);

#if 0
    // Expand the found planes to encompass any connected points that lie near
    // the same plane.
    PlaneComparator<pcl::PointXYZ>::Ptr plane_check(
        new PlaneComparator<pcl::PointXYZ>);
    plane_check->setInputCloud(lores_cloud);
    plane_check->setDistanceThreshold(ctx->cluster_tolerance);

    pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ, pcl::Label>
        plane_expander(plane_check);
    plane_expander.setInputCloud(lores_cloud);

    pcl::PointCloud<pcl::Label> expander_labels;
    std::vector<pcl::PointIndices> expander_label_indices;
    for (unsigned i = 0; i < plane_coeffs.size(); ++i) {
        std::vector<float> &values = plane_coeffs[i].values;
        Eigen::Vector4f &centroid = plane_centroids[i];
        Eigen::Vector4f plane(values[0], values[1], values[2], values[3]);

        plane_check->setPlaneCoefficients(plane);
        expander_labels.points.clear();
        expander_label_indices.clear();
        plane_expander.segment(expander_labels, expander_label_indices);

        if (expander_label_indices.empty()) {
            continue;
        }

        // Find the plane cluster that lies nearest the cluster detected via
        // connected normals
        float distance = FLT_MAX;
        pcl::PointIndices &closest_plane_indices = expander_label_indices[0];
        for (std::vector<pcl::PointIndices>::iterator it =
             expander_label_indices.begin();
             it != expander_label_indices.end(); ++it) {
            Eigen::Vector4f candidate_centroid;
            pcl::compute3DCentroid(*lores_cloud, *it, candidate_centroid);
            float candidate_distance = (candidate_centroid - centroid).norm();
            if (candidate_distance < distance) {
                distance = candidate_distance;
                closest_plane_indices = *it;
            }
        }

        // Discard plane cluster
        for (unsigned p = 0; p < closest_plane_indices.indices.size(); ++p) {
            int idx = closest_plane_indices.indices[p];
            lores_cloud->points[idx] = invalid_pt;
        }
    }
#else
    // Remove the points that make up the planes we found above
    for (std::vector<pcl::PointIndices>::iterator it = plane_indices.begin();
         it != plane_indices.end(); ++it) {
        pcl::PointIndices &indices = *it;
        for (unsigned i = 0; i < indices.indices.size(); ++i) {
            lores_cloud->points[indices.indices[i]] = invalid_pt;
        }
    }
#endif

    end = get_time();
    duration = end - start;
    LOGI("Plane removal (%d planes) took %.3f%s\n",
         (int)plane_indices.size(),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    start = get_time();

    // Use depth clustering to split the cloud into possible human clusters.
    DepthComparator<pcl::PointXYZ, pcl::Normal>::Ptr depth_cluster(
        new DepthComparator<pcl::PointXYZ, pcl::Normal>);
    depth_cluster->setInputCloud(lores_cloud);
    depth_cluster->setInputNormals(tracking->normals);
    depth_cluster->setDepthThreshold(ctx->cluster_tolerance);

    tracking->cluster_labels =
        pcl::PointCloud<pcl::Label>::Ptr(new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ, pcl::Label>
        depth_connector(depth_cluster);
    depth_connector.setInputCloud(lores_cloud);
    depth_connector.segment(*tracking->cluster_labels, cluster_indices);

    end = get_time();
    duration = end - start;
    LOGI("Clustering took (%d clusters) %.3f%s\n",
         (int)cluster_indices.size(),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Assume the largest cluster that has roughly human dimensions and
    // contains its centroid may be a person.
    start = get_time();

    //const float centroid_tolerance = 0.1f;
    std::vector<pcl::PointIndices> persons;
    for (std::vector<pcl::PointIndices>::const_iterator point_it =
         cluster_indices.begin();
         point_it != cluster_indices.end(); ++point_it) {

        // Check if the cluster has human-ish dimensions
        Eigen::Vector4f min, max;
        pcl::getMinMax3D(*lores_cloud, *point_it, min, max);
        Eigen::Vector4f diff = max - min;
        // TODO: Make these limits configurable
        if (diff[0] < 0.15f || diff[0] > 2.0f ||
            diff[1] < 0.8f || diff[1] > 2.45f ||
            diff[2] < 0.05f || diff[2] > 1.5f) {
            continue;
        }
        LOGI("Cluster with %d points, (%.2fx%.2fx%.2f)\n",
             (int)(*point_it).indices.size(), diff[0], diff[1], diff[2]);

#if 0
        // Work out the centroid of the cloud and see if there's a point
        // near there. A human, unless they're falling, ought to contain
        // their center of gravity. If they're jumping or falling, we can
        // probably interpolate joint positions.
        // Note that I guess humans are actually quite frequently in a state
        // of semi-falling, so we have a pretty generous tolerance.
        Eigen::VectorXf centroid;
        pcl::computeNDCentroid(*lores_cloud, *point_it, centroid);

        // Reproject this point into the depth buffer space to get an offset
        // and check if the point exists in the dense cloud.
        int x = (int)
            ((centroid[0] * tracking->depth_camera_intrinsics.fx / centroid[2]) +
             tracking->depth_camera_intrinsics.cx);
        if (x < 0 || x >= (int)tracking->depth_camera_intrinsics.width) {
            continue;
        }

        int y = (int)
            ((centroid[0] * tracking->depth_camera_intrinsics.fy / centroid[2]) +
             tracking->depth_camera_intrinsics.cy);
        if (y < 0 || y >= (int)tracking->depth_camera_intrinsics.height) {
            continue;
        }

        int off = y * tracking->depth_camera_intrinsics.width + x;
        if (std::isnan(hires_cloud->points[off].z) ||
            fabsf(centroid[2] - hires_cloud->points[off].z) >
            centroid_tolerance) {
            continue;
        }
#endif

        persons.push_back(*point_it);
    }

    end = get_time();
    duration = end - start;
    LOGI("People detection took %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    if (persons.size() == 0) {
        // TODO: We should do an interpolation step here.
        LOGE("Skipping detection: Could not find a person cluster\n");
        return false;
    }

    if (tracking->depth_camera_intrinsics.width == 0 ||
        tracking->depth_camera_intrinsics.height == 0)
    {
        LOGE("Skipping detection: depth camera intrinsics uninitialized\n");
        return false;
    }

    start = get_time();
    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;

    if (width == 0 || height == 0) {
        LOGE("Skipping detection: training camera intrinsics uninitialized\n");
        return false;
    }

    // Using the lores point cloud as a mask, reproject the hires point cloud
    // into training camera space.
    glm::mat3 rotate;
    glm::vec3 translate;
    if (tracking->extrinsics_set) {
        const float *r = tracking->depth_to_video_extrinsics.rotation;
        rotate = glm::mat3(r[0], r[1], r[2],
                           r[3], r[4], r[5],
                           r[6], r[7], r[8]);
        const float *t = tracking->depth_to_video_extrinsics.translation;
        translate = glm::vec3(t[0], t[1], t[2]);
    } else {
        rotate = glm::mat3();
        translate = glm::vec3();
    }

    std::vector<float*> depth_images;
    for (std::vector<pcl::PointIndices>::iterator p_it = persons.begin();
         p_it != persons.end(); ++p_it) {

        float *depth_img = (float *)xmalloc(width * height * sizeof(float));
        for (int i = 0; i < width * height; ++i) {
            depth_img[i] = HUGE_DEPTH;
        }

        for (std::vector<int>::const_iterator it = (*p_it).indices.begin();
             it != (*p_it).indices.end (); ++it) {
            int lx = (*it) % lores_cloud->width;
            int ly = (*it) / lores_cloud->width;
            for (int hy = (int)(ly * ctx->seg_res), ey = 0;
                 hy < (int)hires_cloud->height && ey < ctx->seg_res;
                 ++hy, ++ey) {
                for (int hx = (int)(lx * ctx->seg_res), ex = 0;
                     hx < (int)hires_cloud->width && ex < ctx->seg_res;
                     ++hx, ++ex) {
                    int off = hy * hires_cloud->width + hx;

                    // Reproject this point into training camera space
                    glm::vec3 point_t(hires_cloud->points[off].x,
                                      hires_cloud->points[off].y,
                                      hires_cloud->points[off].z);

                    if (tracking->extrinsics_set) {
                        point_t = (rotate * point_t) + translate;
                    }

                    int x = (int)
                        ((point_t.x * tracking->training_camera_intrinsics.fx /
                          point_t.z) + tracking->training_camera_intrinsics.cx);

                    if (x < 0 || x >= width) {
                        continue;
                    }

                    int y = (int)
                        ((point_t.y * tracking->training_camera_intrinsics.fy /
                          point_t.z) + tracking->training_camera_intrinsics.cy);

                    if (y < 0 || y >= height) {
                        continue;
                    }

                    int doff = width * y + x;
                    depth_img[doff] = point_t.z;
                }
            }
        }

        depth_images.push_back(depth_img);
    }


    end = get_time();
    duration = end - start;
    LOGI("Re-projecting %d %dx%d point clouds took %.3f%s\n",
         (int)persons.size(), (int)width, (int)height,
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    float vfov =  pcl::rad2deg(2.0f * atanf(0.5 * height /
                               tracking->training_camera_intrinsics.fy));
    float *weights = (float*)
        xmalloc(width * height * ctx->n_joints * sizeof(float));
    float *label_probs = (float*)xmalloc(width * height * ctx->n_labels *
                                         sizeof(float));
    tracking->skeleton.distance = FLT_MAX;
    for (std::vector<float*>::iterator it = depth_images.begin();
         it != depth_images.end(); ++it) {
        start = get_time();
        float *depth_img = *it;
        infer_labels<float>(ctx->decision_trees, ctx->n_decision_trees,
                            depth_img, width, height, label_probs);
        end = get_time();
        duration = end - start;
        LOGI("Label probability (%d trees, %dx%d) inference took %.3f%s\n",
             (int)ctx->n_decision_trees, (int)width, (int)height,
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));

        start = get_time();
        calc_pixel_weights<float>(depth_img, label_probs, width, height,
                                  ctx->n_labels, ctx->joint_map, weights);
        end = get_time();
        duration = end - start;
        LOGI("Calculating pixel weights took %.3f%s\n",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));

        start = get_time();
        InferredJoints *candidate =
            infer_joints_fast<float>(depth_img, label_probs, weights,
                                     width, height, ctx->n_labels,
                                     ctx->joint_map,
                                     vfov, ctx->joint_params->joint_params);
        xfree(depth_img);

        assert(candidate->n_joints == ctx->n_joints);

        end = get_time();
        duration = end - start;

        struct gm_skeleton candidate_skeleton(ctx->n_joints);
        build_skeleton(ctx, candidate, candidate_skeleton);
        refine_skeleton(ctx, candidate, candidate_skeleton);

        free_joints(candidate);

        LOGI("Inferring joints and refinement took %.3f%s "
             "(confidence: %f, distance: %f)\n",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration),
             candidate_skeleton.confidence, candidate_skeleton.distance);

        if (compare_skeletons(candidate_skeleton, tracking->skeleton)) {
            std::swap(tracking->skeleton, candidate_skeleton);
            std::swap(tracking->label_probs, label_probs);
        }
    }
    xfree(label_probs);
    xfree(weights);

    if (tracking->skeleton.confidence < ctx->skeleton_min_confidence ||
        tracking->skeleton.distance > ctx->skeleton_max_distance) {
        return false;
    }

    start = get_time();

    // TODO: We just take the most confident skeleton above, but we should
    //       probably establish some thresholds and spit out multiple skeletons.
    if (ctx->bone_sanitisation) {
        sanitise_skeleton(ctx, tracking->skeleton, tracking->frame->timestamp);
    } else {
        build_bones(ctx, tracking->skeleton);
    }
    for (int j = 0; j < ctx->n_joints; j++) {
        int idx = j * 3;
        tracking->joints_processed[idx] = tracking->skeleton.joints[j].x;
        tracking->joints_processed[idx+1] = tracking->skeleton.joints[j].y;
        tracking->joints_processed[idx+2] = tracking->skeleton.joints[j].z;
    }

    end = get_time();
    duration = end - start;
    LOGI("Joint processing took %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    return true;
}

static struct gm_event *
event_alloc(enum gm_event_type type)
{
    struct gm_event *event =
        (struct gm_event *)xcalloc(sizeof(struct gm_event), 1);

    event->type = type;

    return event;
}

void
gm_context_event_free(struct gm_event *event)
{
    free(event);
}

static void
request_frame(struct gm_context *ctx)
{
    struct gm_event *event = event_alloc(GM_EVENT_REQUEST_FRAME);

    event->request_frame.flags =
        GM_REQUEST_FRAME_DEPTH | GM_REQUEST_FRAME_VIDEO;

    ctx->event_callback(ctx, event, ctx->callback_data);
}

static void
notify_tracking(struct gm_context *ctx)
{
    struct gm_event *event = event_alloc(GM_EVENT_TRACKING_READY);

    ctx->event_callback(ctx, event, ctx->callback_data);
}

void
update_face_detect_luminance_buffer(struct gm_context *ctx,
                                    struct gm_tracking_impl *tracking,
                                    enum gm_format format,
                                    uint8_t *video)
{
    int width = tracking->video_camera_intrinsics.width;
    int height = tracking->video_camera_intrinsics.height;

    if (!ctx->grey_width) {
        ctx->grey_width = width;
        ctx->grey_height = height;

#ifndef DOWNSAMPLE_ON_GPU
#ifdef DOWNSAMPLE_1_4
        ctx->grey_buffer_1_2.resize((width / 2) * (height / 2));
#endif
#endif
    }

    glimpse::wrapped_image<unsigned char> orig_grey_img;

    switch(format) {
    case GM_FORMAT_RGB_U8:
        foreach_xy_off(width, height) {
            tracking->face_detect_buf[off] = RGB2Y(video[off * 3],
                                                   video[off * 3 + 1],
                                                   video[off * 3 + 2]);
        }
        break;
    case GM_FORMAT_RGBX_U8:
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 4];
            uint8_t g = video[off * 4 + 1];
            uint8_t b = video[off * 4 + 2];
            tracking->face_detect_buf[off] = RGB2Y(r, g, b);
        }
        break;
    case GM_FORMAT_RGBA_U8:
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 4];
            uint8_t g = video[off * 4 + 1];
            uint8_t b = video[off * 4 + 2];
            tracking->face_detect_buf[off] = RGB2Y(r, g, b);
        }
        break;
    case GM_FORMAT_LUMINANCE_U8:
        memcpy(tracking->face_detect_buf, video, width * height);
        break;
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_Z_U16_MM:
    case GM_FORMAT_Z_F32_M:
    case GM_FORMAT_Z_F16_M:
    case GM_FORMAT_POINTS_XYZC_F32_M:
        gm_assert(ctx->log, 0, "Unexpected format for video buffer");
        return;
    }

    orig_grey_img.wrap(ctx->grey_width,
                       ctx->grey_height,
                       ctx->grey_width, //stride
                       static_cast<unsigned char *>(tracking->face_detect_buf));

#ifndef DOWNSAMPLE_ON_GPU
    uint64_t start, end, duration_ns;

#ifdef DOWNSAMPLE_1_2
#ifdef DOWNSAMPLE_1_4
    /* 1/4 resolution */
    glimpse::wrapped_image<unsigned char> grey_1_2_img;
    grey_1_2_img.wrap(width / 2,
                      height / 2,
                      width / 2, //stride
                      static_cast<unsigned char *>(ctx->grey_buffer_1_2.data()));

    glimpse::wrapped_image<unsigned char> grey_1_4_img;
    grey_1_4_img.wrap(width / 4,
                      height / 4,
                      width / 4, //stride
                      static_cast<unsigned char *>(tracking->face_detect_buf));
#else
    /* half resolution */
    glimpse::wrapped_image<unsigned char> grey_1_2_img;
    grey_1_2_img.wrap(width / 2,
                      height / 2,
                      width / 2, //stride
                      static_cast<unsigned char *>(tracking->face_detect_buf));
#endif
#endif

#ifdef DOWNSAMPLE_1_2
    LOGI("Started resizing frame");
    start = get_time();
    dlib::resize_image(orig_grey_img, grey_1_2_img,
                       dlib::interpolate_bilinear());
    end = get_time();
    duration_ns = end - start;
    LOGI("Frame scaled to 1/2 size on CPU in %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

#ifdef DOWNSAMPLE_1_4
    start = get_time();
    dlib::resize_image(grey_1_2_img, grey_1_4_img,
                       dlib::interpolate_bilinear());
    end = get_time();
    duration_ns = end - start;

    LOGI("Frame scaled to 1/4 size on CPU in %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));
#endif // DOWNSAMPLE_1_4
#endif // DOWNSAMPLE_1_2


#endif // !DOWNSAMPLE_ON_GPU
}

static void
copy_and_rotate_depth_buffer(struct gm_context *ctx,
                             struct gm_tracking_impl *tracking,
                             enum gm_format format,
                             struct gm_buffer *buffer)
{
    int width = ctx->basis_depth_camera_intrinsics.width;
    int height = ctx->basis_depth_camera_intrinsics.height;
    int rot_width = tracking->depth_camera_intrinsics.width;
    int rot_height = tracking->depth_camera_intrinsics.height;
    enum gm_rotation rotation = tracking->frame->camera_rotation;
    void *depth = buffer->data;
    float *depth_copy = tracking->depth;

    int num_points;

    // Not ideal how we use `with_rotated_rx_ry_roff` per-pixel, but it lets
    // us easily combine our rotation with our copy...
    //
    // XXX: it could be worth reading multiple scanlines at a time so we could
    // write out cache lines at a time instead of only 4 bytes (for rotated
    // images).
    //
    switch (format) {
    case GM_FORMAT_Z_U16_MM:
        foreach_xy_off(width, height) {
            float depth_m = ((uint16_t *)depth)[off] / 1000.f;
            with_rotated_rx_ry_roff(x, y, width, height,
                                    rotation, rot_width,
                                    { depth_copy[roff] = depth_m; });
        }
        break;
    case GM_FORMAT_Z_F32_M:
        foreach_xy_off(width, height) {
            float depth_m = ((float *)depth)[off];
            with_rotated_rx_ry_roff(x, y, width, height,
                                    rotation, rot_width,
                                    { depth_copy[roff] = depth_m; });
        }
        break;
    case GM_FORMAT_Z_F16_M:
        foreach_xy_off(width, height) {
            float depth_m = ((half *)depth)[off];
            with_rotated_rx_ry_roff(x, y, width, height,
                                    rotation, rot_width,
                                    { depth_copy[roff] = depth_m; });
        }
        break;
    case GM_FORMAT_POINTS_XYZC_F32_M: {

        /* XXX: Tango doesn't give us a 2D depth buffer which we would prefer
         * and we can't be sure that the re-projected point cloud will fill our
         * 2D depth buffer 100% so we're forced to clear it first too :(
         */
        memset(depth_copy, 0, width * height * sizeof(float));

        num_points = buffer->len / 16;

        float fx = ctx->basis_depth_camera_intrinsics.fx;
        float fy = ctx->basis_depth_camera_intrinsics.fy;
        float cx = ctx->basis_depth_camera_intrinsics.cx;
        float cy = ctx->basis_depth_camera_intrinsics.cy;

        float k1, k2, k3;

        /* XXX: we only support applying the brown's model... */
        bool apply_distortion = ctx->apply_depth_distortion;
        if (apply_distortion) {
            switch (ctx->basis_depth_camera_intrinsics.distortion_model) {
            case GM_DISTORTION_NONE:
                apply_distortion = false;
                break;
            case GM_DISTORTION_FOV_MODEL:
                apply_distortion = false;
                break;
            case GM_DISTORTION_BROWN_K1_K2:
                k1 = ctx->basis_depth_camera_intrinsics.distortion[0];
                k2 = ctx->basis_depth_camera_intrinsics.distortion[1];
                k3 = 0;
                break;
            case GM_DISTORTION_BROWN_K1_K2_K3:
                k1 = ctx->basis_depth_camera_intrinsics.distortion[0];
                k2 = ctx->basis_depth_camera_intrinsics.distortion[1];
                k3 = ctx->basis_depth_camera_intrinsics.distortion[2];
                break;
            case GM_DISTORTION_BROWN_K1_K2_P1_P2_K3:
                k1 = ctx->basis_depth_camera_intrinsics.distortion[0];
                k2 = ctx->basis_depth_camera_intrinsics.distortion[1];
                k3 = ctx->basis_depth_camera_intrinsics.distortion[4];
                /* Ignoring tangential distortion */
                break;
            }
        }

        for (int off = 0; off < num_points; off++) {
            float *xyzc = ((float *)buffer->data) + 4 * off;
            float ru, ru2, ru3, ru4, ru5, ru6, ru7;
            float rd;

            // Reproject this point into training camera space
            glm::vec3 point_t(xyzc[0], xyzc[1], xyzc[2]);

            int x;
            if (apply_distortion) {
                ru = sqrtf((point_t.x*point_t.x + point_t.y*point_t.y) /
                                 (point_t.z*point_t.z));
                ru2 = ru*ru;
                ru3 = ru2*ru;
                ru4 = ru3*ru;
                ru5 = ru4*ru;
                ru6 = ru5*ru;
                ru7 = ru6*ru;

                // Google documented their POLY_3 distortion model should
                // be evaluated as:
                //   rd = ru + k1 * ru^3 + k2 * ru^5 + k3 * ru^7
                // they also refer to the same model as Brown's with only
                // k1, k2 and k3 coefficients, but e.g. referencing Wikipedia
                // and looking for other interpretations of the Brown-Conrady
                // model then it looks like there's some inconsistency with
                // the ru exponents used. Wikipedia uses:
                //   k1 * ru^2 + k2 * ru^4 + k3 * ru^6
                rd = ru + k1 * ru3 + k2 * ru5 + k3 * ru7;

                x = (int)(point_t.x / point_t.z * fx * rd / ru + cx);
            } else
                x = (int)((point_t.x * fx / point_t.z) + cx);

            if (x < 0 || x >= width) {
                continue;
            }

            int y;
            if (apply_distortion)
                y = (int)(point_t.y / point_t.z * fy * rd / ru + cy);
            else
                y = (int)((point_t.y * fy / point_t.z) + cy);
            if (y < 0 || y >= height) {
                continue;
            }

            with_rotated_rx_ry_roff(x, y, width, height,
                                    rotation, rot_width,
                                    { depth_copy[roff] = point_t.z; });
        }

        if (ctx->depth_gap_fill) {
            // Our code doesn't deal well with gaps in the data, so make some
            // effort to fill in gaps that may have been caused by bad data,
            // distortion transforms, etc.
            struct blank {
                int off;
                float new_depth;
            };
            std::vector<struct blank> blanks;

            foreach_xy_off(rot_width, rot_height) {
                if (std::isnormal(depth_copy[off])) {
                    continue;
                }

                // Check to see if gaps can be filled by surrounding pixels on
                // either the horizontal or vertical axis
                int left = 0, right = 0, up = 0, down = 0;
                for (int i = 1;
                     i <= ctx->gap_dist && !(left && right) && !(up && down); ++i) {
                    if (!left && (x - i) >= 0 && std::isnormal(depth_copy[off-i])) {
                        left = i;
                    }
                    if (!right && (x + i) < rot_width &&
                        std::isnormal(depth_copy[off+i])) {
                        right = i;
                    }
                    if (!up && (y - i) >= 0 &&
                        std::isnormal(depth_copy[off - (i * rot_width)])) {
                        up = i;
                    }
                    if (!down && (y + i) < rot_height &&
                        std::isnormal(depth_copy[off + (i * rot_width)])) {
                        down = i;
                    }
                }

                bool hvalid = false;
                if (left && right) {
                    float lval = depth_copy[off-left];
                    float rval = depth_copy[off+right];
                    float distance = fabsf(rval - lval);
                    if (distance < ctx->gap_tolerance * (left + right)) {
                        hvalid = true;
                    }
                }
                bool vvalid = false;
                if (up && down) {
                    int pix_dist = up + down;
                    float uval = depth_copy[off-(up * rot_width)];
                    float dval = depth_copy[off+(down * rot_width)];
                    float distance = fabsf(dval - uval);
                    if (distance < ctx->gap_tolerance * pix_dist) {
                        if (pix_dist < left + right || !(left && right)) {
                            vvalid = true;
                            hvalid = false;
                        }
                    }
                }

                if (hvalid || vvalid) {
                    // Note, this interpolation isn't correct because it doesn't
                    // take depth into account. This has a more extreme effect
                    // at oblique angles, but hopefully we won't be filling in
                    // too many pixels so it shouldn't matter too much.
                    if (hvalid) {
                        int pix_dist = left + right;
                        float lval = depth_copy[off-left];
                        float rval = depth_copy[off+right];
                        for (int o = -left + 1, i = 0; o < right; ++o, ++i) {
                            float new_depth = ((lval * ((pix_dist - 1) - o)) +
                                               (rval * o)) / (float)(pix_dist - 1);
                            blanks.push_back({off + o, new_depth});
                        }
                    } else {
                        int pix_dist = up + down;
                        float uval = depth_copy[off-(up * rot_width)];
                        float dval = depth_copy[off+(down * rot_width)];
                        for (int o = -up + 1, i = 0; o < down; ++o, ++i) {
                            float new_depth = ((uval * ((pix_dist - 1) - o)) +
                                               (dval * o)) / (float)(pix_dist - 1);
                            blanks.push_back({off + (o * rot_width), new_depth});
                        }
                    }
                }
            }

            for (unsigned i = 0; i < blanks.size(); ++i) {
                depth_copy[blanks[i].off] = blanks[i].new_depth;
            }
        }
        break;
    }
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_LUMINANCE_U8:
    case GM_FORMAT_RGB_U8:
    case GM_FORMAT_RGBX_U8:
    case GM_FORMAT_RGBA_U8:
        gm_assert(ctx->log, 0, "Unexpected format for depth buffer");
        break;
    }
}

static struct gm_tracking_impl *
mem_pool_acquire_tracking(struct gm_mem_pool *pool)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)
        mem_pool_acquire_resource(pool);
    tracking->base.ref = 1;

    tracking->success = false;

    return tracking;
}

static struct gm_prediction_impl *
mem_pool_acquire_prediction(struct gm_mem_pool *pool)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)
        mem_pool_acquire_resource(pool);
    prediction->base.ref = 1;
    return prediction;
}

void
gm_context_rotate_intrinsics(struct gm_context *ctx,
                             const struct gm_intrinsics *intrinsics_in,
                             struct gm_intrinsics *intrinsics_out,
                             enum gm_rotation rotation)
{
    switch (rotation) {
    case GM_ROTATION_0:
        *intrinsics_out = *intrinsics_in;
        break;
    case GM_ROTATION_90:
        intrinsics_out->width = intrinsics_in->height;
        intrinsics_out->height = intrinsics_in->width;

        intrinsics_out->cx = intrinsics_in->cy;
        intrinsics_out->cy = intrinsics_in->width - intrinsics_in->cx;

        intrinsics_out->fx = intrinsics_in->fy;
        intrinsics_out->fy = intrinsics_in->fx;
        break;
    case GM_ROTATION_180:
        intrinsics_out->width = intrinsics_in->width;
        intrinsics_out->height = intrinsics_in->height;

        intrinsics_out->cx = intrinsics_in->width - intrinsics_in->cx;
        intrinsics_out->cy = intrinsics_in->height - intrinsics_in->cy;

        intrinsics_out->fx = intrinsics_in->fx;
        intrinsics_out->fy = intrinsics_in->fy;
        break;
    case GM_ROTATION_270:
        intrinsics_out->width = intrinsics_in->height;
        intrinsics_out->height = intrinsics_in->width;

        intrinsics_out->cx = intrinsics_in->height - intrinsics_in->cy;
        intrinsics_out->cy = intrinsics_in->cx;

        intrinsics_out->fx = intrinsics_in->fy;
        intrinsics_out->fy = intrinsics_in->fx;
        break;
    }
}

static void *
detector_thread_cb(void *data)
{
    struct gm_context *ctx = (struct gm_context *)data;

    gm_debug(ctx->log, "Started Glimpse tracking thread");

    uint64_t start = get_time();
    ctx->detector = dlib::get_frontal_face_detector();
    uint64_t end = get_time();
    uint64_t duration = end - start;

    gm_debug(ctx->log, "Initialized Dlib frontal face detector: %.3f%s",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));

    //LOGI("Dropped all but the first (front-facing HOG) from the DLib face detector");
    //ctx->detector.w.resize(1);

    //LOGI("Detector debug %p", &ctx->detector.scanner);

    char *err = NULL;
    struct gm_asset *predictor_asset =
        gm_asset_open(ctx->log,
                      "shape_predictor_68_face_landmarks.dat",
                      GM_ASSET_MODE_BUFFER,
                      &err);
    if (predictor_asset) {
        const void *buf = gm_asset_get_buffer(predictor_asset);
        off_t len = gm_asset_get_length(predictor_asset);
        std::istringstream stream_in(std::string((char *)buf, len));
        try {
            dlib::deserialize(ctx->face_feature_detector, stream_in);
        } catch (dlib::serialization_error &e) {
            gm_warn(ctx->log, "Failed to deserialize shape predictor: %s", e.info.c_str());
        }

        gm_debug(ctx->log, "Mapped shape predictor asset %p, len = %d", buf, (int)len);
        gm_asset_close(predictor_asset);
    } else {
        gm_warn(ctx->log, "Failed to open shape predictor asset: %s", err);
        free(err);
    }

    while (!ctx->stopping) {
        struct gm_frame *frame = NULL;

        LOGI("Waiting for new frame to start tracking\n");
        pthread_mutex_lock(&ctx->frame_ready_mutex);
        while (!ctx->frame_ready && !ctx->stopping) {
            pthread_cond_wait(&ctx->frame_ready_cond, &ctx->frame_ready_mutex);
        }
        frame = ctx->frame_ready;
        ctx->frame_ready = NULL;
        pthread_mutex_unlock(&ctx->frame_ready_mutex);

        if (ctx->stopping) {
            gm_debug(ctx->log, "Stopping tracking after frame acquire (context being destroyed)");
            if (frame)
                gm_frame_unref(frame);
            break;
        }

        start = get_time();
        gm_debug(ctx->log, "Starting tracking iteration (%" PRIu64 ")\n",
                 frame->timestamp);

        struct gm_tracking_impl *tracking =
            mem_pool_acquire_tracking(ctx->tracking_pool);

        tracking->frame = frame;

        /* FIXME: rotate the camera extrinsics according to the display rotation */
        tracking->extrinsics_set = ctx->basis_extrinsics_set;
        tracking->depth_to_video_extrinsics = ctx->basis_depth_to_video_extrinsics;

        gm_context_rotate_intrinsics(ctx,
                                     &ctx->basis_video_camera_intrinsics,
                                     &tracking->video_camera_intrinsics,
                                     tracking->frame->camera_rotation);
        gm_context_rotate_intrinsics(ctx,
                                     &ctx->basis_depth_camera_intrinsics,
                                     &tracking->depth_camera_intrinsics,
                                     tracking->frame->camera_rotation);

        tracking->training_camera_intrinsics = ctx->basis_training_camera_intrinsics;

        copy_and_rotate_depth_buffer(ctx,
                                     tracking,
                                     frame->depth_format,
                                     frame->depth);

        /* FIXME: re-enable support for face detection */
#if 0
        /* While downsampling on the CPU we currently do that synchronously
         * when we are notified of a new frame.
         */
#ifdef DOWNSAMPLE_ON_GPU
        gm_debug(ctx->log, "Waiting for new scaled frame for face detection");
        pthread_mutex_lock(&ctx->scaled_frame_cond_mutex);
        ctx->need_new_scaled_frame = true;
        while (ctx->need_new_scaled_frame && !ctx->stopping) {
            pthread_cond_wait(&ctx->scaled_frame_available_cond,
                              &ctx->scaled_frame_cond_mutex);
        }
        pthread_mutex_unlock(&ctx->scaled_frame_cond_mutex);

        if (ctx->stopping) {
            gm_debug(ctx->log, "Stopping tracking after frame downsample (context being destroyed)");
            gm_tracking_unref(tracking);
            break;
        }
#else
        update_face_detect_luminance_buffer(ctx,
                                            tracking,
                                            frame->video_format,
                                            (uint8_t *)frame->video->data);

#endif

        gm_context_detect_faces(ctx, tracking);
#endif

        bool tracked = gm_context_track_skeleton(ctx, tracking);

        end = get_time();
        duration = end - start;
        gm_debug(ctx->log, "Finished skeletal tracking (%.3f%s)",
                 get_duration_ns_print_scale(duration),
                 get_duration_ns_print_scale_suffix(duration));

        pthread_mutex_lock(&ctx->tracking_swap_mutex);

        if (tracked) {
            tracking->success = true;

            // Clear the tracking history if we've gone back in time
            if (ctx->n_tracking && tracking->frame->timestamp <
                ctx->tracking_history[0]->frame->timestamp) {
                gm_warn(ctx->log,
                        "Tracking has gone back in time, clearing history");

                for (int i = 0; i < ctx->n_tracking; ++i) {
                    gm_tracking_unref(&ctx->tracking_history[i]->base);
                    ctx->tracking_history[i] = NULL;
                }
                ctx->n_tracking = 0;
            }

            for (int i = TRACK_FRAMES - 1; i > 0; i--)
                std::swap(ctx->tracking_history[i], ctx->tracking_history[i - 1]);
            if (ctx->tracking_history[0]) {
                gm_debug(ctx->log, "pushing %p out of tracking history fifo (ref = %d)\n",
                         ctx->tracking_history[0],
                         ctx->tracking_history[0]->base.ref);
                gm_tracking_unref(&ctx->tracking_history[0]->base);
            }
            ctx->tracking_history[0] = (struct gm_tracking_impl *)
                gm_tracking_ref(&tracking->base);

            gm_debug(ctx->log, "adding %p to tracking history fifo (ref = %d)\n",
                     ctx->tracking_history[0],
                     ctx->tracking_history[0]->base.ref);

            if (ctx->n_tracking < TRACK_FRAMES)
                ctx->n_tracking++;

            gm_debug(ctx->log, "tracking history len = %d:", ctx->n_tracking);
            for (int i = 0; i < ctx->n_tracking; i++) {
                gm_debug(ctx->log, "%d) %p (ref = %d)", i,
                         ctx->tracking_history[i],
                         ctx->tracking_history[i]->base.ref);
            }
        }

        /* Hold onto the latest tracking regardless of whether it was
         * successful so that a user can still access the all the information
         * related to tracking.
         */
        if (ctx->latest_tracking)
            gm_tracking_unref(&ctx->latest_tracking->base);
        ctx->latest_tracking = tracking;

        pthread_mutex_unlock(&ctx->tracking_swap_mutex);

        notify_tracking(ctx);

        gm_debug(ctx->log, "Requesting new frame for skeletal tracking");
        /* We throttle frame acquisition according to our tracking rate... */
        request_frame(ctx);
    }

    return NULL;
}

static void
tracking_state_free(struct gm_mem_pool *pool,
                    void *self,
                    void *user_data)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)self;

    free(tracking->label_probs);
    free(tracking->joints_processed);

    free(tracking->depth);

    free(tracking->face_detect_buf);

    if (tracking->frame) {
        gm_frame_unref(tracking->frame);
    }

    delete tracking;
}

static void
tracking_state_recycle(struct gm_tracking *self)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)self;
    struct gm_mem_pool *pool = tracking->pool;

    gm_assert(tracking->ctx->log, tracking->base.ref == 0,
              "Unbalanced tracking unref");

    gm_frame_unref(tracking->frame);
    tracking->frame = NULL;

    tracking->trail.clear();

    mem_pool_recycle_resource(pool, tracking);
}

static void
tracking_add_breadcrumb(struct gm_tracking *self, const char *tag)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)self;
    struct trail_crumb crumb;

    gm_assert(tracking->ctx->log, tracking->base.ref >= 0,
              "Use of frame after free");

    snprintf(crumb.tag, sizeof(crumb.tag), "%s", tag);

    crumb.n_frames = gm_backtrace(crumb.backtrace_frame_pointers,
                                  1, // skip top stack frame
                                  10);

    pthread_mutex_lock(&tracking->trail_lock);
    tracking->trail.push_back(crumb);
    pthread_mutex_unlock(&tracking->trail_lock);
}

static void *
tracking_state_alloc(struct gm_mem_pool *pool, void *user_data)
{
    struct gm_context *ctx = (struct gm_context *)user_data;
    struct gm_tracking_impl *tracking = new gm_tracking_impl();

    tracking->base.ref = 1;
    tracking->base.api = &tracking->vtable;

    tracking->vtable.free = tracking_state_recycle;
    tracking->vtable.add_breadcrumb = tracking_add_breadcrumb;

    tracking->pool = pool;
    tracking->ctx = ctx;

    int labels_width = ctx->basis_training_camera_intrinsics.width;
    int labels_height = ctx->basis_training_camera_intrinsics.height;

    assert(labels_width);
    assert(labels_height);

    tracking->label_probs = (float *)xcalloc(labels_width *
                                             labels_height *
                                             ctx->n_labels, sizeof(float));

    tracking->skeleton.joints.resize(ctx->n_joints);
    tracking->skeleton.bones.resize(ctx->n_joints);
    tracking->joints_processed = (float *)
      xcalloc(ctx->n_joints, 3 * sizeof(float));

    int depth_width = ctx->basis_depth_camera_intrinsics.width;
    int depth_height = ctx->basis_depth_camera_intrinsics.height;

    assert(depth_width);
    assert(depth_height);

    tracking->depth = (float *)
      xcalloc(depth_width * depth_height, sizeof(float));

    int video_width = ctx->basis_video_camera_intrinsics.width;
    int video_height = ctx->basis_video_camera_intrinsics.height;

    tracking->face_detect_buf =
        (uint8_t *)xcalloc(video_width * video_height, 1);

#ifdef DOWNSAMPLE_1_2
#ifdef DOWNSAMPLE_1_4
    tracking->face_detect_buf_width = video_width / 4;
    tracking->face_detect_buf_height = video_height / 4;
#else
    tracking->face_detect_buf_width = video_width / 2;
    tracking->face_detect_buf_height = video_height / 2;
#endif
#else
    tracking->face_detect_buf_width = video_width;
    tracking->face_detect_buf_height = video_height;
#endif

    return tracking;
}

static void
prediction_free(struct gm_mem_pool *pool,
                void *self,
                void *user_data)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)self;

    for (int i = 0; i < prediction->n_tracking; ++i) {
        gm_tracking_unref((struct gm_tracking *)
                          prediction->tracking_history[i]);
    }

    delete prediction;
}

static void
prediction_recycle(struct gm_prediction *self)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)self;
    struct gm_mem_pool *pool = prediction->pool;

    gm_assert(prediction->ctx->log, prediction->base.ref == 0,
              "Unbalanced prediction unref");

    for (int i = 0; i < prediction->n_tracking; ++i) {
        gm_tracking_unref((struct gm_tracking *)
                          prediction->tracking_history[i]);
    }
    prediction->n_tracking = 0;

    prediction->trail.clear();

    mem_pool_recycle_resource(pool, prediction);
}

static void
prediction_add_breadcrumb(struct gm_prediction *self, const char *tag)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)self;
    struct trail_crumb crumb;

    gm_assert(prediction->ctx->log, prediction->base.ref >= 0,
              "Use of frame after free");

    snprintf(crumb.tag, sizeof(crumb.tag), "%s", tag);

    crumb.n_frames = gm_backtrace(crumb.backtrace_frame_pointers,
                                  1, // skip top stack frame
                                  10);

    pthread_mutex_lock(&prediction->trail_lock);
    prediction->trail.push_back(crumb);
    pthread_mutex_unlock(&prediction->trail_lock);
}

static void *
prediction_alloc(struct gm_mem_pool *pool, void *user_data)
{
    struct gm_context *ctx = (struct gm_context *)user_data;
    struct gm_prediction_impl *prediction = new gm_prediction_impl();

    prediction->base.ref = 1;
    prediction->base.api = &prediction->vtable;

    prediction->vtable.free = prediction_recycle;
    prediction->vtable.add_breadcrumb = prediction_add_breadcrumb;

    prediction->pool = pool;

    prediction->ctx = ctx;

    return (void *)prediction;
}

static void
print_trail_for(struct gm_logger *log, void *object, std::vector<struct trail_crumb> *trail)
{
    gm_debug(log, "Trail for %p:", object);

    for (unsigned i = 0; i < trail->size(); i++) {
        struct trail_crumb crumb = trail->at(i);
        if (crumb.n_frames) {
            struct gm_backtrace backtrace = {
                crumb.n_frames,
                (const void **)crumb.backtrace_frame_pointers
            };
            int line_len = 100;
            char *formatted = (char *)alloca(crumb.n_frames * line_len);

            gm_debug(log, "%d) tag = %s", i, crumb.tag);
            gm_logger_get_backtrace_strings(log, &backtrace,
                                            line_len, (char *)formatted);
            for (int i = 0; i < crumb.n_frames; i++) {
                char *line = formatted + line_len * i;
                gm_debug(log, "   #%i %s", i, line);
            }
        }
    }
}

static void
print_tracking_info_cb(struct gm_mem_pool *pool,
                       void *resource,
                       void *user_data)
{
    struct gm_context *ctx = (struct gm_context *)user_data;
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)resource;

    gm_assert(ctx->log, tracking != NULL, "Spurious NULL tracking resource");
    gm_error(ctx->log, "Unreleased tracking object %p, ref count = %d, paper trail len = %d",
             tracking,
             tracking->base.ref,
             (int)tracking->trail.size());

    if (tracking->trail.size())
        print_trail_for(ctx->log, tracking, &tracking->trail);
}

static void
gm_context_clear_tracking(struct gm_context *ctx)
{
    if (ctx->latest_tracking) {
        gm_tracking_unref(&ctx->latest_tracking->base);
        ctx->latest_tracking = NULL;
    }
    for (int i = 0; i < ctx->n_tracking; ++i) {
        gm_tracking_unref(&ctx->tracking_history[i]->base);
        ctx->tracking_history[i] = NULL;
    }
    ctx->n_tracking = 0;

    mem_pool_foreach(ctx->tracking_pool,
                     print_tracking_info_cb,
                     ctx);
    mem_pool_free_resources(ctx->tracking_pool);
    mem_pool_free_resources(ctx->prediction_pool);
}

static int
gm_context_start_tracking(struct gm_context *ctx, char **err)
{
    /* XXX: maybe make it an explicit, public api to start running detection
     */
    int ret = pthread_create(&ctx->detect_thread,
                             nullptr, /* default attributes */
                             detector_thread_cb,
                             ctx);

    if (ret == 0) {
        pthread_setname_np(ctx->detect_thread, "Glimpse Track");
    }

    return ret;
}

static void
gm_context_stop_tracking(struct gm_context *ctx)
{
    gm_debug(ctx->log, "stopping context tracking");

    pthread_mutex_lock(&ctx->liveness_lock);

    /* The tracking thread checks for this, such that we can now expect it to
     * exit within a finite amount of time.
     */
    ctx->stopping = true;

    /* Note: we intentionally don't keep this lock for the duration of
     * destruction because the assumption is that we only need to wait for
     * the render hook or other entrypoints that were already running to
     * finish the caller should otherwise ensure no further calls are made.
     * Dropping the lock asap increases the chance of our debug assertions
     * recognising any mistake made.
     */
    pthread_mutex_unlock(&ctx->liveness_lock);

    /* It's possible the tracker thread is waiting for a new frame in
     * pthread_cond_wait, and we don't want it to wait indefinitely...
     */
    pthread_mutex_lock(&ctx->frame_ready_mutex);
    pthread_cond_signal(&ctx->frame_ready_cond);
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    /* It's also possible the tracker thread is waiting for a downsampled
     * frame in pthread_cond_wait...
     */
    pthread_mutex_lock(&ctx->scaled_frame_cond_mutex);
    pthread_cond_signal(&ctx->scaled_frame_available_cond);
    pthread_mutex_unlock(&ctx->scaled_frame_cond_mutex);

    if (ctx->detect_thread) {
        void *tracking_retval = NULL;
        int ret = pthread_join(ctx->detect_thread, &tracking_retval);
        if (ret < 0) {
            gm_error(ctx->log, "Failed waiting for tracking thread to complete: %s",
                     strerror(ret));
        } else {
            ctx->detect_thread = 0;
        }

        if (tracking_retval != 0) {
            gm_error(ctx->log, "Tracking thread exited with value = %d",
                     (int)(intptr_t)tracking_retval);
        }
    }
}

void
gm_context_destroy(struct gm_context *ctx)
{
    /* XXX: The context may be accessed asynchronously in the following ways:
     *
     *  1) The tracking thread
     *  2) The render thread hook
     *  3) Various thread safe getter or notify entry-points
     *
     * We created/own the tracking thread and so we have to stop this thread
     * anyway after which we don't have to worry about it accessing ctx state.
     *
     * The render thread hook will take the liveness_lock and assert
     * ctx->destroying == false before accessing the ctx state, so assuming the
     * render hook always completes in a finite time we can simply wait until
     * we've acquired this lock before starting to destroy the context. Note:
     * the hook can use an assertion to check ctx->destroying because it's the
     * callers responsibility to ensure that the render hook doesn't continue
     * to be called with a context pointer that's about to become invalid. I.e.
     * We're only concerned about render hooks that are already running when
     * gm_context_stop_tracking() is called.
     *
     * Any other thread-safe entry-points should take the liveness_lock and
     * assert ctx->destroying == false. We should expect that the caller has
     * ensured that these functions will no longer be called because the caller
     * knows that the context pointer is about to become invalid, but should
     * also scatter some assertions to try and catch spurious notifications
     * during destruction.
     */

    pthread_mutex_lock(&ctx->liveness_lock);

    ctx->destroying = true;

    /* Note: we intentionally don't keep this lock for the duration of
     * destruction because the assumption is that we only need to wait for
     * the render hook or other entrypoints that were already running to
     * finish. The caller should otherwise ensure no further calls are made.
     * Dropping the lock asap increases the chance of our debug assertions
     * recognising any mistake made.
     */
    pthread_mutex_unlock(&ctx->liveness_lock);

    gm_context_stop_tracking(ctx);

    /* Free the prediction pool. The user must have made sure to unref any
     * predictions before destroying the context.
     */
    mem_pool_free(ctx->prediction_pool);

    /* Make sure all resourced are returned to their pools before destroying
     * the pools which will in-turn destroy the resources...
     */
    gm_context_clear_tracking(ctx);
    mem_pool_foreach(ctx->tracking_pool,
                     print_tracking_info_cb,
                     ctx);
    mem_pool_free(ctx->tracking_pool);

    /* Only taking the mutex for the sake of a debug assertion within
     * gm_context_notify_frame() (to double check that we don't see any
     * notifications during destruction)...
     */
    pthread_mutex_lock(&ctx->frame_ready_mutex);
    if (ctx->frame_ready) {
        gm_frame_unref(ctx->frame_ready);
        ctx->frame_ready = NULL;
    }
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    free(ctx->depth_color_stops);
    free(ctx->heat_color_stops);

    for (int i = 0; i < ctx->n_decision_trees; i++)
        free_tree(ctx->decision_trees[i]);
    xfree(ctx->decision_trees);

    if (ctx->joint_params)
        free_jip(ctx->joint_params);

    if (ctx->joint_map)
        json_value_free(ctx->joint_map);

    if (ctx->joint_stats) {
        for (int i = 0; i < ctx->n_joints; i++) {
            xfree(ctx->joint_stats[i].connections);
            xfree(ctx->joint_stats[i].dist);
        }
        xfree(ctx->joint_stats);
    }

    delete ctx;
}

struct gm_context *
gm_context_new(struct gm_logger *logger, char **err)
{
    /* NB: we can't just calloc this struct since it contains C++ class members
     * that need to be constructed appropriately
     */
    struct gm_context *ctx = new gm_context();

    ctx->log = logger;

    pthread_cond_init(&ctx->skel_track_cond, NULL);
    pthread_mutex_init(&ctx->skel_track_cond_mutex, NULL);

    ctx->tracking_pool = mem_pool_alloc(logger,
                                        "tracking",
                                        INT_MAX, // max size
                                        tracking_state_alloc,
                                        tracking_state_free,
                                        ctx); // user data

    ctx->prediction_pool = mem_pool_alloc(logger,
                                          "prediction",
                                          INT_MAX,
                                          prediction_alloc,
                                          prediction_free,
                                          ctx);

    /* Load the decision trees immediately so we know how many labels we're
     * dealing with asap.
     */
    int max_trees = 3;
    ctx->n_decision_trees = 0;
    ctx->decision_trees = (RDTree**)xcalloc(max_trees, sizeof(RDTree*));

    for (int i = 0; i < max_trees; i++) {
        char rdt_name[16];
        char json_name[16];
        char *name = NULL;

        xsnprintf(rdt_name, sizeof(rdt_name), "tree%u.rdt", i);
        xsnprintf(json_name, sizeof(json_name), "tree%u.json", i);

        char *catch_err = NULL;
        struct gm_asset *tree_asset = gm_asset_open(logger,
                                                    rdt_name,
                                                    GM_ASSET_MODE_BUFFER,
                                                    &catch_err);
        if (tree_asset) {
            name = rdt_name;
            ctx->decision_trees[i] =
                load_tree((uint8_t *)gm_asset_get_buffer(tree_asset),
                          gm_asset_get_length(tree_asset));
        } else {
            free(catch_err);
            char *open_err = NULL;

            name = json_name;
            tree_asset = gm_asset_open(logger,
                                       json_name,
                                       GM_ASSET_MODE_BUFFER,
                                       &open_err);
            if (!tree_asset) {
                gm_warn(logger,
                         "Failed to open tree%u.rdt and tree%u.json: %s",
                         i, i, open_err);
                free(open_err);
                break;
            }

            /* XXX: Technically we should pass a NUL terminated string but
             * since we're assuming we're passing a valid Json Object then we
             * can rely on parsing terminating on the closing '}' without
             * depending on finding a terminating NUL. Otherwise we would
             * have to copy the asset into a larger buffer so we can
             * explicitly add the NUL.
             */
            ctx->decision_trees[i] =
                load_json_tree((uint8_t *)gm_asset_get_buffer(tree_asset),
                               gm_asset_get_length(tree_asset));
        }

        gm_asset_close(tree_asset);

        if (!ctx->decision_trees[i]) {
            gm_warn(logger, "Failed to load %s", name);
            break;
        }

        ctx->n_decision_trees++;
    }

    if (!ctx->n_decision_trees) {
        gm_throw(logger, err, "Failed to open any decision tree assets");
        gm_context_destroy(ctx);
        return NULL;
    }

    ctx->n_labels = ctx->decision_trees[0]->header.n_labels;

    int ret = gm_context_start_tracking(ctx, err);
    if (ret != 0) {
        gm_throw(logger, err,
                 "Failed to start face detector thread: %s", strerror(ret));
        gm_context_destroy(ctx);
    }

    int labels_width = 172;
    int labels_height = 224;
    ctx->basis_training_camera_intrinsics.width = labels_width;
    ctx->basis_training_camera_intrinsics.height = labels_height;
    ctx->basis_training_camera_intrinsics.cx = 86;
    ctx->basis_training_camera_intrinsics.cy = 112;
    ctx->basis_training_camera_intrinsics.fx = 217.461437772;
    ctx->basis_training_camera_intrinsics.fy = 217.461437772;

    ctx->joint_map = NULL;
    char *open_err = NULL;
    struct gm_asset *joint_map_asset = gm_asset_open(logger,
                                                     "joint-map.json",
                                                     GM_ASSET_MODE_BUFFER,
                                                     &open_err);
    if (joint_map_asset) {
        const void *buf = gm_asset_get_buffer(joint_map_asset);
        unsigned len = gm_asset_get_length(joint_map_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length and expects a NUL terminated string...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';

        ctx->joint_map = json_parse_string(js_string);

        xfree(js_string);
        gm_asset_close(joint_map_asset);

        if (!ctx->joint_map) {
            gm_throw(logger, err, "Failed to open joint map\n");
            gm_context_destroy(ctx);
            return NULL;
        }

        ctx->n_joints = json_array_get_count(json_array(ctx->joint_map));

    } else {
        gm_throw(logger, err, "Failed to open joint-map.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }

    ctx->joint_params = NULL;

    struct gm_asset *joint_params_asset =
        gm_asset_open(logger,
                      "joint-params.json",
                      GM_ASSET_MODE_BUFFER,
                      &open_err);
    if (joint_params_asset) {
        const void *buf = gm_asset_get_buffer(joint_params_asset);
        unsigned len = gm_asset_get_length(joint_params_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';

        JSON_Value *root = json_parse_string(js_string);
        if (root) {
            ctx->joint_params = joint_params_from_json(root);
            json_value_free(root);
        }

        xfree(js_string);
        gm_asset_close(joint_params_asset);

        if (!ctx->joint_params) {
            gm_throw(logger, err, "Failed to laod joint params from json");
            gm_context_destroy(ctx);
            return NULL;
        }
    } else {
        gm_throw(logger, err, "Failed to open joint-params.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }

    // Load joint statistics for improving the quality of predicted joint
    // positions.
    ctx->joint_stats = NULL;

    struct gm_asset *joint_stats_asset =
        gm_asset_open(logger,
                      "joint-dist.json", GM_ASSET_MODE_BUFFER, &open_err);
    if (joint_stats_asset) {
        ctx->joint_stats = (struct joint_info *)
            xcalloc(ctx->n_joints, sizeof(struct joint_info));
        for (int i = 0; i < ctx->n_joints; i++) {
            ctx->joint_stats[i].connections = (int *)
                xmalloc(ctx->n_joints * sizeof(int));
            ctx->joint_stats[i].dist = (struct joint_dist *)
                xmalloc(ctx->n_joints * sizeof(struct joint_dist));
        }

        // Discover joint connections
        for (int i = 0; i < ctx->n_joints; i++) {
            JSON_Object *joint =
                json_array_get_object(json_array(ctx->joint_map), i);
            JSON_Array *connections =
                json_object_get_array(joint, "connections");
            for (int c = 0; c < (int)json_array_get_count(connections); c++) {
                const char *name = json_array_get_string(connections, c);
                for (int j = 0; j < ctx->n_joints; j++) {
                    JSON_Object *connection =
                        json_array_get_object(json_array(ctx->joint_map), j);
                    if (strcmp(json_object_get_string(connection, "joint"),
                               name) != 0) { continue; }

                    // Add the connection to this joint and add the reverse
                    // connection.
                    int idx = ctx->joint_stats[i].n_connections;
                    ctx->joint_stats[i].connections[idx] = j;
                    ++ctx->joint_stats[i].n_connections;

                    idx = ctx->joint_stats[j].n_connections;
                    ctx->joint_stats[j].connections[idx] = i;
                    ++ctx->joint_stats[j].n_connections;

                    break;
                }
            }
        }

        const void *buf = gm_asset_get_buffer(joint_stats_asset);
        JSON_Value *json = json_parse_string((char *)buf);

        assert((int)json_array_get_count(json_array(json)) == ctx->n_joints);
        for (int i = 0; i < ctx->n_joints; i++) {
            JSON_Array *stats = json_array_get_array(json_array(json), i);
            assert((int)json_array_get_count(stats) == ctx->n_joints);

            for (int j = 0; j < ctx->n_joints; j++) {
                JSON_Object *stat = json_array_get_object(stats, j);
                ctx->joint_stats[i].dist[j].min = (float)
                    json_object_get_number(stat, "min");
                ctx->joint_stats[i].dist[j].mean = (float)
                    json_object_get_number(stat, "mean");
                ctx->joint_stats[i].dist[j].max = (float)
                    json_object_get_number(stat, "max");
            }
        }

        gm_asset_close(joint_stats_asset);
        json_value_free(json);
    } else {
        gm_throw(logger, err, "Failed to open joint-dist.json: %s", open_err);
        free(open_err);

        // We can continue without the joint stats asset, just results may be
        // poorer quality.
    }

    ctx->depth_color_stops_range = 5; // meters
    alloc_rgb_color_stops(&ctx->depth_color_stops,
                          &ctx->n_depth_color_stops,
                          depth_rainbow,
                          ARRAY_LEN(depth_rainbow),
                          ctx->depth_color_stops_range,
                          0.25); // steps

    ctx->heat_color_stops_range = 1; // normalised probability
    alloc_rgb_color_stops(&ctx->heat_color_stops,
                          &ctx->n_heat_color_stops,
                          heat_map_rainbow,
                          ARRAY_LEN(heat_map_rainbow),
                          ctx->heat_color_stops_range, // range
                          1.f / ARRAY_LEN(heat_map_rainbow)); // step

    struct gm_ui_property prop;

    ctx->gap_dist = 2;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "gap_dist";
    prop.desc = "The distance in 2d pixels at which to look at neighbouring "
                "pixels to fill in gaps when input data is a point cloud.";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->gap_dist;
    prop.int_state.min = 0;
    prop.int_state.max = 5;
    ctx->properties.push_back(prop);

    ctx->gap_tolerance = 0.05f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "gap_tolerance";
    prop.desc = "Distance threshold when interpolating between gaps";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->gap_tolerance;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.2f;
    ctx->properties.push_back(prop);

    ctx->depth_gap_fill = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "depth_gap_fill";
    prop.desc = "Fill small gaps in the depth buffers";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->depth_gap_fill;
    ctx->properties.push_back(prop);

    ctx->apply_depth_distortion = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "apply_depth_distortion";
    prop.desc = "Apply the distortion model of depth camera";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->apply_depth_distortion;
    ctx->properties.push_back(prop);

    ctx->cloud_res = 1;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "cloud_res";
    prop.desc = "Resolution divider for depth camera cloud";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->cloud_res;
    prop.int_state.min = 1;
    prop.int_state.max = 4;
    ctx->properties.push_back(prop);

    ctx->min_depth = 0.5;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "min_depth";
    prop.desc = "throw away points nearer than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->min_depth;
    prop.float_state.min = 0.0;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->max_depth = 3.5;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_depth";
    prop.desc = "throw away points further than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->max_depth;
    prop.float_state.min = 0.5;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->seg_res = 3;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "seg_res";
    prop.desc = "Resolution divider for running human segmentation";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->seg_res;
    prop.int_state.min = 1;
    prop.int_state.max = 4;
    ctx->properties.push_back(prop);

    ctx->normal_depth_change = 0.03f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "normal_depth_change";
    prop.desc = "Maximum depth change for computing object borders";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->normal_depth_change;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.5f;
    ctx->properties.push_back(prop);

    ctx->normal_smooth = 3.0f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "normal_smooth";
    prop.desc = "Size of the area used to smooth normals";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->normal_smooth;
    prop.float_state.min = 1.0f;
    prop.float_state.max = 10.f;
    ctx->properties.push_back(prop);

    ctx->min_inliers = 50;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "min_inliers";
    prop.desc = "Minimum number of inliers when doing plane segmentation";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->min_inliers;
    prop.int_state.min = 5;
    prop.int_state.max = 200;
    ctx->properties.push_back(prop);

    ctx->angular_threshold = 10.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "angular_threshold";
    prop.desc = "Angular threshold for multi-plane segmentation";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->angular_threshold;
    prop.float_state.min = 0.1f;
    prop.float_state.max = 45.f;
    ctx->properties.push_back(prop);

    ctx->distance_threshold = 0.03f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "distance_threshold";
    prop.desc = "Distance threshold for multi-plane segmentation";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->distance_threshold;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.5f;
    ctx->properties.push_back(prop);

    ctx->refinement_distance_threshold = 0.15f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "refinement_distance_threshold";
    prop.desc = "Distance threshold for multi-plane segmentation refinement";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->refinement_distance_threshold;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.5f;
    ctx->properties.push_back(prop);

    ctx->max_curvature = 0.001f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_curvature";
    prop.desc = "Maximum curvature of a plane for multi-plane segmentation";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->max_curvature;
    prop.float_state.min = 0.0005f;
    prop.float_state.max = 0.03f;
    ctx->properties.push_back(prop);

    ctx->cluster_tolerance = 0.03f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "cluster_tolerance";
    prop.desc = "Distance threshold when clustering points";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->cluster_tolerance;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.2f;
    ctx->properties.push_back(prop);

    ctx->joint_refinement = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_refinement";
    prop.desc = "Favour less confident joint predictions that conform "
                "to a statistical joint position model better";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->joint_refinement;
    ctx->properties.push_back(prop);

    ctx->max_joint_predictions = 4;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_joint_predictions";
    prop.desc = "Maximum number of absolute joint position predictions to make "
                "across tracking history";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &ctx->max_joint_predictions;
    prop.int_state.min = 0;
    prop.int_state.max = TRACK_FRAMES - 1;
    ctx->properties.push_back(prop);

    ctx->max_prediction_delta = 100.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_prediction_delta";
    prop.desc = "Maximum time to predict from a tracking frame (in ms)";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->max_prediction_delta;
    prop.float_state.min = 0.f;
    prop.float_state.max = 1000.f;
    ctx->properties.push_back(prop);

    ctx->prediction_decay = 1.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "prediction_decay";
    prop.desc = "Prediction time decay. A multiplier for max_prediction_delta "
                "where time after that point travels increasingly slowly to "
                "dampen predictions.";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->prediction_decay;
    prop.float_state.min = 0.f;
    prop.float_state.max = 4.f;
    ctx->properties.push_back(prop);

    ctx->bone_sanitisation = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bone_sanitisation";
    prop.desc = "Calculate bone lengths and angles and try to smooth out "
                "unexpected changes.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->bone_sanitisation;
    ctx->properties.push_back(prop);

    ctx->bone_length_variance = 0.05f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bone_length_variance";
    prop.desc = "Maximum allowed variance of bone length between frames";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->bone_length_variance;
    prop.float_state.min = 0.f;
    prop.float_state.max = 0.2f;
    ctx->properties.push_back(prop);

    ctx->bone_rotation_variance = 360.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bone_rotation_variance";
    prop.desc = "Maximum allowed rotation of a bone (degrees/s)";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->bone_rotation_variance;
    prop.float_state.min = 100.f;
    prop.float_state.max = 1500.f;
    ctx->properties.push_back(prop);

    ctx->joint_move_threshold = 1.5f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_move_threshold";
    prop.desc = "Minimum travel distance (m/s) "
                "before considering joint prediction";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->joint_move_threshold;
    prop.float_state.min = 0.3f;
    prop.float_state.max = 3.0f;
    ctx->properties.push_back(prop);

    ctx->joint_max_travel = 3.0f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_max_travel";
    prop.desc = "Maximum travel distance (m/s) "
                "before considering joint prediction";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->joint_max_travel;
    prop.float_state.min = 1.0f;
    prop.float_state.max = 5.0f;
    ctx->properties.push_back(prop);

    ctx->joint_scale_threshold = 2.5f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_scale_threshold";
    prop.desc = "Maximum growth difference (x/s) "
                "before considering joint prediction";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->joint_scale_threshold;
    prop.float_state.min = 1.5f;
    prop.float_state.max = 20.f;
    ctx->properties.push_back(prop);

    ctx->skeleton_min_confidence = 1000.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "skeleton_min_skeleton";
    prop.desc = "Minimum cumulative joint confidence of a skeleton";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->skeleton_min_confidence;
    prop.float_state.min = 50.f;
    prop.float_state.max = 5000.f;
    ctx->properties.push_back(prop);

    ctx->skeleton_max_distance = 0.2f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "skeleton_max_distance";
    prop.desc = "Maximum cumulative square distance from min/max testing "
                "bone lengths";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->skeleton_max_distance;
    prop.float_state.min = 0.01f;
    prop.float_state.max = 0.5f;
    ctx->properties.push_back(prop);

    ctx->debug_label = -1;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "debug_label";
    prop.desc = "visualize specific label probabilities";
    prop.type = GM_PROPERTY_INT;
    prop.enum_state.ptr = &ctx->debug_label;

    struct gm_ui_enumerant enumerant;
    enumerant = gm_ui_enumerant();
    enumerant.name = "most likely";
    enumerant.desc = "Visualize Most Probable Labels";
    prop.type = GM_PROPERTY_ENUM;
    enumerant.val = -1;
    ctx->label_enumerants.push_back(enumerant);

    gm_assert(ctx->log,
              ctx->n_labels == ARRAY_LEN(label_names),
              "Mismatched label name count");

    for (int i = 0; i < ctx->n_labels; i++) {
        enumerant = gm_ui_enumerant();
        enumerant.name = label_names[i];
        enumerant.desc = label_names[i];
        enumerant.val = i;
        ctx->label_enumerants.push_back(enumerant);
    }
    prop.enum_state.n_enumerants = ctx->label_enumerants.size();
    prop.enum_state.enumerants = ctx->label_enumerants.data();
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    return ctx;
}

void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics)
{
    ctx->basis_depth_camera_intrinsics = *intrinsics;
}

void
gm_context_set_video_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics)
{
    ctx->basis_video_camera_intrinsics = *intrinsics;
}

void
gm_context_set_depth_to_video_camera_extrinsics(struct gm_context *ctx,
                                                struct gm_extrinsics *extrinsics)
{
    if (extrinsics) {
        ctx->basis_depth_to_video_extrinsics = *extrinsics;
        ctx->basis_extrinsics_set = true;
    } else {
        ctx->basis_extrinsics_set = false;
    }
}

const gm_intrinsics *
gm_context_get_training_intrinsics(struct gm_context *ctx)
{
    return &ctx->basis_training_camera_intrinsics;
}

const gm_intrinsics *
gm_tracking_get_video_camera_intrinsics(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return &tracking->video_camera_intrinsics;
}

const gm_intrinsics *
gm_tracking_get_depth_camera_intrinsics(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return &tracking->depth_camera_intrinsics;
}

const gm_intrinsics *
gm_tracking_get_training_camera_intrinsics(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return &tracking->training_camera_intrinsics;
}

const float *
gm_tracking_get_depth(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    return tracking->depth;
}

const float *
gm_tracking_get_joint_positions(struct gm_tracking *_tracking,
                                int *n_joints)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    if (n_joints) *n_joints = tracking->ctx->n_joints;
    return tracking->joints_processed;
}

bool
gm_tracking_has_skeleton(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return tracking->success;
}

const struct gm_skeleton *
gm_tracking_get_skeleton(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return tracking->success ? &tracking->skeleton : NULL;
}

void
gm_tracking_create_rgb_label_map(struct gm_tracking *_tracking,
                                 int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    uint8_t n_labels = ctx->n_labels;

    *width = (int)ctx->basis_training_camera_intrinsics.width;
    *height = (int)ctx->basis_training_camera_intrinsics.height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    gm_assert(ctx->log, ctx->debug_label < n_labels,
              "Can't create RGB map of invalid label %u",
              ctx->debug_label);

    foreach_xy_off(*width, *height) {
        uint8_t label = 0;
        float pr = -1.0;
        float *pr_table = &tracking->label_probs[off * n_labels];
        for (uint8_t l = 0; l < n_labels; l++) {
            if (pr_table[l] > pr) {
                label = l;
                pr = pr_table[l];
            }
        }

        uint8_t r;
        uint8_t g;
        uint8_t b;

        if (ctx->debug_label == -1) {
            r = default_palette[label].red;
            g = default_palette[label].green;
            b = default_palette[label].blue;
        } else {
            struct color col = stops_color_from_val(ctx->heat_color_stops,
                                                    ctx->n_heat_color_stops,
                                                    1,
                                                    pr_table[ctx->debug_label]);
            r = col.r;
            g = col.g;
            b = col.b;
        }

        (*output)[off * 3] = r;
        (*output)[off * 3 + 1] = g;
        (*output)[off * 3 + 2] = b;
    }
}

void
gm_tracking_create_rgb_depth(struct gm_tracking *_tracking,
                             int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    *width = (int)tracking->depth_camera_intrinsics.width;
    *height = (int)tracking->depth_camera_intrinsics.height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        float depth = tracking->depth[off];
        struct color rgb = stops_color_from_val(ctx->depth_color_stops,
                                                ctx->n_depth_color_stops,
                                                ctx->depth_color_stops_range,
                                                depth);
        (*output)[off * 3] = rgb.r;
        (*output)[off * 3 + 1] = rgb.g;
        (*output)[off * 3 + 2] = rgb.b;
    }
}

void
gm_tracking_create_rgb_video(struct gm_tracking *_tracking,
                             int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    *width = (int)ctx->basis_video_camera_intrinsics.width;
    *height = (int)ctx->basis_video_camera_intrinsics.height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    int rot_width = tracking->video_camera_intrinsics.width;
    enum gm_format format = tracking->frame->video_format;
    enum gm_rotation rotation = tracking->frame->camera_rotation;
    uint8_t *video = (uint8_t *)tracking->frame->video->data;

    // Not ideal how we use `with_rotated_rx_ry_roff` per-pixel, but it lets
    // us easily combine our rotation with our copy...
    //
    // XXX: it could be worth reading multiple scanlines at a time so we could
    // write out cache lines at a time instead of only 4 bytes (for rotated
    // images).
    //
    switch(format) {
    case GM_FORMAT_RGB_U8:
        foreach_xy_off(*width, *height) {
            with_rotated_rx_ry_roff(x, y, *width, *height,
                                    rotation, rot_width,
                                    {
                                        (*output)[roff*3] = video[off*3];
                                        (*output)[roff*3+1] = video[off*3+1];
                                        (*output)[roff*3+2] = video[off*3+2];
                                    });
        }
        break;
    case GM_FORMAT_RGBX_U8:
    case GM_FORMAT_RGBA_U8:
        foreach_xy_off(*width, *height) {
            with_rotated_rx_ry_roff(x, y, *width, *height,
                                    rotation, rot_width,
                                    {
                                        (*output)[roff*3] = video[off*4];
                                        (*output)[roff*3+1] = video[off*4+1];
                                        (*output)[roff*3+2] = video[off*4+2];
                                    });
        }
        break;
    case GM_FORMAT_LUMINANCE_U8:
        foreach_xy_off(*width, *height) {
            with_rotated_rx_ry_roff(x, y, *width, *height,
                                    rotation, rot_width,
                                    {
                                        uint8_t lum = video[off];
                                        (*output)[roff*3] = lum;
                                        (*output)[roff*3+1] = lum;
                                        (*output)[roff*3+2] = lum;
                                    });
        }
        break;
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_Z_U16_MM:
    case GM_FORMAT_Z_F32_M:
    case GM_FORMAT_Z_F16_M:
    case GM_FORMAT_POINTS_XYZC_F32_M:
        gm_assert(ctx->log, 0, "Unexpected format for video buffer");
        return;
    }

    // Output is rotated, so make sure output width/height are correct
    if (rotation == GM_ROTATION_90 || rotation == GM_ROTATION_270) {
        std::swap(*width, *height);
    }
}

void
gm_tracking_create_rgb_normals(struct gm_tracking *_tracking,
                               int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    //struct gm_context *ctx = tracking->ctx;

    *width = (int)tracking->normals->width;
    *height = (int)tracking->normals->height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        pcl::Normal &norm = tracking->normals->points[off];
        if (std::isnan(norm.normal_z)) {
            (*output)[off * 3] = 0;
            (*output)[off * 3 + 1] = 0;
            (*output)[off * 3 + 2] = 0;
        } else {
            uint8_t r = (uint8_t)((norm.normal_x * 127) + 127);
            uint8_t g = (uint8_t)((norm.normal_y * 127) + 127);
            uint8_t b = (uint8_t)(255 - ((norm.normal_z * 127) + 127));
            (*output)[off * 3] = r;
            (*output)[off * 3 + 1] = g;
            (*output)[off * 3 + 2] = b;
        }
    }
}

void
gm_tracking_create_rgb_normal_clusters(struct gm_tracking *_tracking,
                                       int *width, int *height,
                                       uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    //struct gm_context *ctx = tracking->ctx;

    *width = (int)tracking->normal_labels->width;
    *height = (int)tracking->normal_labels->height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        int label = tracking->normal_labels->points[off].label;
        png_color *color =
            &default_palette[label % ARRAY_LEN(default_palette)];
        float shade = 1.f - (float)(label / ARRAY_LEN(default_palette)) / 30.f;
        (*output)[off * 3] = (uint8_t)(color->red * shade);
        (*output)[off * 3 + 1] = (uint8_t)(color->green * shade);
        (*output)[off * 3 + 2] = (uint8_t)(color->blue * shade);
    }
}

void
gm_tracking_create_rgb_candidate_clusters(struct gm_tracking *_tracking,
                                          int *width, int *height,
                                          uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    //struct gm_context *ctx = tracking->ctx;

    *width = (int)tracking->cluster_labels->width;
    *height = (int)tracking->cluster_labels->height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        int label = tracking->cluster_labels->points[off].label;
        png_color *color =
            &default_palette[label % ARRAY_LEN(default_palette)];
        float shade = 1.f - (float)(label / ARRAY_LEN(default_palette)) / 10.f;
        (*output)[off * 3] = (uint8_t)(color->red * shade);
        (*output)[off * 3 + 1] = (uint8_t)(color->green * shade);
        (*output)[off * 3 + 2] = (uint8_t)(color->blue * shade);
    }
}

const float *
gm_tracking_get_label_probabilities(struct gm_tracking *_tracking,
                                    int *width,
                                    int *height)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    *width = tracking->training_camera_intrinsics.width;
    *height = tracking->training_camera_intrinsics.height;

    return tracking->label_probs;
}

uint64_t
gm_tracking_get_timestamp(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    return tracking->frame->timestamp;
}

int
gm_skeleton_get_n_joints(const struct gm_skeleton *skeleton)
{
    return (int)skeleton->joints.size();
}

float
gm_skeleton_get_confidence(const struct gm_skeleton *skeleton)
{
    return skeleton->confidence;
}

float
gm_skeleton_get_distance(const struct gm_skeleton *skeleton)
{
    return skeleton->distance;
}

const struct gm_joint *
gm_skeleton_get_joint(const struct gm_skeleton *skeleton, int joint)
{
    return &skeleton->joints[joint];
}

/* Note this may be called via any arbitrary thread
 */
bool
gm_context_notify_frame(struct gm_context *ctx,
                        struct gm_frame *frame)
{
    /* Ignore frames without depth and video data */
    if (frame->depth == NULL || frame->video == NULL)
        return false;

    pthread_mutex_lock(&ctx->liveness_lock);

    gm_debug(ctx->log, "gm_context_notify_frame: frame = %p, depth=%p, video=%p",
             frame, frame->depth, frame->video);

    gm_assert(ctx->log, !ctx->destroying,
              "Spurious notification during tracking context destruction");

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    gm_assert(ctx->log, !ctx->destroying, "Spurious frame notification during destruction");
    if (ctx->frame_ready)
        gm_frame_unref(ctx->frame_ready);
    ctx->frame_ready = gm_frame_ref(frame);
    pthread_cond_signal(&ctx->frame_ready_cond);
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    pthread_mutex_unlock(&ctx->liveness_lock);

    return true;
}

void
gm_context_flush(struct gm_context *ctx, char **err)
{
    gm_context_stop_tracking(ctx);

    gm_context_clear_tracking(ctx);

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    if (ctx->frame_ready) {
        gm_frame_unref(ctx->frame_ready);
        ctx->frame_ready = NULL;
    }
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    ctx->stopping = false;
    gm_debug(ctx->log, "Glimpse context flushed, restarting tracking thread");

    int ret = gm_context_start_tracking(ctx, NULL);
    if (ret != 0) {
        gm_throw(ctx->log, err,
                 "Failed to start face detector thread: %s", strerror(ret));
    }
}

struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx)
{
    struct gm_tracking *tracking = NULL;

    pthread_mutex_lock(&ctx->tracking_swap_mutex);
    if (ctx->latest_tracking) {
        tracking = gm_tracking_ref(&ctx->latest_tracking->base);

        gm_debug(ctx->log, "get_latest_tracking = %p (ref = %d)\n",
                 ctx->latest_tracking,
                 ctx->latest_tracking->base.ref);
    }
    pthread_mutex_unlock(&ctx->tracking_swap_mutex);

    return tracking;
}

static int
get_closest_tracking_frame(struct gm_tracking_impl **tracking_history,
                           int n_tracking, uint64_t timestamp)
{
    int closest_frame = 0;
    uint64_t closest_diff = UINT64_MAX;
    for (int i = 0; i < n_tracking; ++i) {
        uint64_t *t1 = &tracking_history[i]->frame->timestamp;
        uint64_t diff = (*t1 > timestamp) ?
            (*t1 - timestamp) : (timestamp - *t1);
        if (diff < closest_diff) {
            closest_diff = diff;
            closest_frame = i;
        } else {
            break;
        }
    }
    return closest_frame;
}

struct gm_prediction *
gm_context_get_prediction(struct gm_context *ctx, uint64_t timestamp)
{
    struct gm_prediction_impl *prediction =
        mem_pool_acquire_prediction(ctx->prediction_pool);

    // Copy the current tracking history from the context
    pthread_mutex_lock(&ctx->tracking_swap_mutex);

    if (!ctx->n_tracking) {
        pthread_mutex_unlock(&ctx->tracking_swap_mutex);
        gm_prediction_unref(&prediction->base);
        return NULL;
    }

    for (int i = 0; i < ctx->n_tracking; ++i) {
        prediction->tracking_history[i] = ctx->tracking_history[i];
        gm_tracking_ref(&prediction->tracking_history[i]->base);
    }
    prediction->n_tracking = ctx->n_tracking;

    pthread_mutex_unlock(&ctx->tracking_swap_mutex);

    // Pre-fill the skeleton with the closest frame
    int closest_frame =
        get_closest_tracking_frame(prediction->tracking_history,
                                   prediction->n_tracking, timestamp);
    struct gm_skeleton &closest_skeleton =
        prediction->tracking_history[closest_frame]->skeleton;
    prediction->skeleton = closest_skeleton;

    // Validate the timestamp
    timestamp = calculate_decayed_timestamp(
        prediction->tracking_history[closest_frame]->frame->timestamp,
        timestamp, ctx->max_prediction_delta, ctx->prediction_decay);
    prediction->timestamp = timestamp;

    int parent_head = 0;
    uint64_t closest_timestamp =
        ctx->tracking_history[closest_frame]->frame->timestamp;
    if (timestamp != closest_timestamp && prediction->n_tracking > 1) {
        // Work out the two nearest frames and the interpolation value
        int h1;
        if (timestamp > closest_timestamp) {
            h1 = (closest_frame == 0) ? 0 : closest_frame - 1;
        } else {
            h1 = (closest_frame == prediction->n_tracking - 1) ?
                closest_frame - 1 : closest_frame;
        }
        int h2 = h1 + 1;

        struct gm_tracking_impl *frame1 = ctx->tracking_history[h1];
        struct gm_tracking_impl *frame2 = ctx->tracking_history[h2];
        float t = (timestamp - frame2->frame->timestamp) /
                  (float)(frame1->frame->timestamp - frame2->frame->timestamp);

        // Use linear interpolation to place the parent bone(s). We'll use
        // the interpolated angles to place the rest of the bones.
        for (std::vector<struct bone_info>::iterator it =
             closest_skeleton.bones.begin();
             it != closest_skeleton.bones.end(); ++it) {
            struct bone_info &bone = *it;
            if (bone.head == parent_head) {
                interpolate_joints(
                    frame2->skeleton.joints[bone.head],
                    frame1->skeleton.joints[bone.head],
                    t, prediction->skeleton.joints[bone.head]);
                interpolate_joints(
                    frame2->skeleton.joints[bone.tail],
                    frame1->skeleton.joints[bone.tail],
                    t, prediction->skeleton.joints[bone.tail]);
            }
        }

        // Interpolate angles for the rest of the bones
        for (std::vector<struct bone_info>::iterator it =
             closest_skeleton.bones.begin();
             it != closest_skeleton.bones.end(); ++it) {
            struct bone_info &bone = *it;

            if (bone.head < 0) {
                continue;
            }

            struct bone_info &parent_bone =
                closest_skeleton.bones[bone.head];
            if (parent_bone.head < 0) {
                continue;
            }

            // Find the angle to rotate the parent bone. Note, we're relying
            // on bones being stored in an order where we can rely on the
            // bone's parent being seen before any descendents.
            glm::mat3 rotate = glm::mat3_cast(
                glm::slerp(frame2->skeleton.bones[bone.tail].angle,
                           frame1->skeleton.bones[bone.tail].angle, t));

            glm::vec3 parent_vec = glm::normalize(
                glm::vec3(prediction->skeleton.joints[parent_bone.tail].x -
                          prediction->skeleton.joints[parent_bone.head].x,
                          prediction->skeleton.joints[parent_bone.tail].y -
                          prediction->skeleton.joints[parent_bone.head].y,
                          prediction->skeleton.joints[parent_bone.tail].z -
                          prediction->skeleton.joints[parent_bone.head].z));
            glm::vec3 new_tail = ((parent_vec * rotate) * bone.length);
            new_tail.x += prediction->skeleton.joints[bone.head].x;
            new_tail.y += prediction->skeleton.joints[bone.head].y;
            new_tail.z += prediction->skeleton.joints[bone.head].z;

            prediction->skeleton.joints[bone.tail].x = new_tail.x;
            prediction->skeleton.joints[bone.tail].y = new_tail.y;
            prediction->skeleton.joints[bone.tail].z = new_tail.z;
        }
    }

    return &prediction->base;
}

uint64_t
gm_prediction_get_timestamp(struct gm_prediction *_prediction)
{
    struct gm_prediction_impl *prediction =
        (struct gm_prediction_impl *)_prediction;
    return prediction->timestamp;
}

const struct gm_skeleton *
gm_prediction_get_skeleton(struct gm_prediction *_prediction)
{
    struct gm_prediction_impl *prediction =
        (struct gm_prediction_impl *)_prediction;
    return &prediction->skeleton;
}

void
gm_context_render_thread_hook(struct gm_context *ctx)
{
    pthread_mutex_lock(&ctx->liveness_lock);

    gm_assert(ctx->log, !ctx->destroying,
              "Spurious render thread hook during tracking context destruction");

    /*
     * FIXME: clean all this stuff up...
     */

    if (!ctx->need_new_scaled_frame) {
        pthread_mutex_unlock(&ctx->liveness_lock);
        return;
    }

    /* FIXME: how can we query the info (namely the downsampling rate) at
     * runtime from the ctx->detector.scanner.pyramid_type, instead of hard
     * coding...
     */
    //dlib::pyramid_down<6> pyr;

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
    struct gm_tracking_impl *tracking = ctx->tracking_front;

    struct {
        float x, y, s, t;
    } quad_strip[] = {
        { -1,  1, 0, 1, }, //  0  2
        { -1, -1, 0, 0, }, //  | /|
        {  1,  1, 1, 1, }, //  |/ |
        {  1, -1, 1, 0  }  //  1  3
    };

    uint64_t start, end, duration_ns;


    LOGI("Downsampling via GLES");

    if (!attrib_quad_rot_scale_bo_) {
        glGenBuffers(1, &attrib_quad_rot_scale_bo_);
        glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_strip), quad_strip, GL_STATIC_DRAW);

        glGenBuffers(1, &attrib_quad_rot_scale_bo_);
        glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);
    }

    /* Our first draw call will combine downsampling and rotation... */
    glBindBuffer(GL_ARRAY_BUFFER, attrib_quad_rot_scale_bo_);

    bool need_portrait_downsample_fb;

    if (display_rotation_ != current_attrib_bo_rotation_) {
        LOGI("Orientation change to account for with face detection");
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

        switch(display_rotation_) {
        case GM_ROTATION_0:
            need_portrait_downsample_fb = true;
            LOGI("> rotation = 0");
            break;
        case GM_ROTATION_90:
            need_portrait_downsample_fb = false;
            LOGI("> rotation = 90");
            break;
        case GM_ROTATION_180:
            need_portrait_downsample_fb = true;
            LOGI("> rotation = 180");
            break;
        case GM_ROTATION_270:
            need_portrait_downsample_fb = false;
            LOGI("> rotation = 270");
            break;
        }
        current_attrib_bo_rotation_ = display_rotation_;
    } else {
        need_portrait_downsample_fb = have_portrait_downsample_fb_;
    }

    long rotated_frame_width, rotated_frame_height;

    if (need_portrait_downsample_fb) {
        rotated_frame_width = ctx->grey_height;
        rotated_frame_height = ctx->grey_width;
    } else {
        rotated_frame_width = ctx->grey_width;
        rotated_frame_height = ctx->grey_height;
    }
    LOGI("rotated frame width = %d, height = %d",
         (int)rotated_frame_width, (int)rotated_frame_height);

    if (need_portrait_downsample_fb != have_portrait_downsample_fb_) {
        if (downsample_fbo_) {
            LOGI("Discarding previous downsample fbo and texture");
            glDeleteFramebuffers(1, &downsample_fbo_);
            downsample_fbo_ = 0;
            glDeleteTextures(1, &downsample_tex2d_);
            downsample_tex2d_ = 0;
        }
        if (ctx->read_back_fbo) {
            LOGI("Discarding previous read_back_fbo and texture");
            glDeleteFramebuffers(1, &ctx->read_back_fbo);
            ctx->read_back_fbo = 0;
        }
    }

    if (!downsample_fbo_) {
        LOGI("Allocating new %dx%d downsample fbo + texture",
             (int)(rotated_frame_width / 2),
             (int)(rotated_frame_height / 2));

        glGenFramebuffers(1, &downsample_fbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, downsample_fbo_);

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

        yuv_frame_scale_program_ = create_program(ctx, vert_shader, frag_shader, NULL);

        ctx->attrib_quad_rot_scale_pos = glGetAttribLocation(yuv_frame_scale_program_, "pos");
        ctx->attrib_quad_rot_scale_tex_coords = glGetAttribLocation(yuv_frame_scale_program_, "tex_coords_in");
        uniform_tex_sampler_ = glGetUniformLocation(yuv_frame_scale_program_, "yuv_tex_sampler");

        glUseProgram(yuv_frame_scale_program_);

        glUniform1i(uniform_tex_sampler_, 0);

        LOGI("Created level0 scale shader");
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

        scale_program_ = create_program(ctx, vert_shader, frag_shader, NULL);

        ctx->attrib_quad_rot_scale_pos =
            glGetAttribLocation(scale_program_, "pos");
        ctx->attrib_quad_rot_scale_tex_coords =
            glGetAttribLocation(scale_program_, "tex_coords_in");
        uniform_tex_sampler_ = glGetUniformLocation(scale_program_, "texture");

        glUseProgram(scale_program_);

        glUniform1i(uniform_tex_sampler_, 0);

        LOGI("Created scale shader");
    }

    if (!ctx->cam_tex) {
        glGenTextures(1, &ctx->cam_tex);
        glBindTexture(GL_TEXTURE_2D, ctx->cam_tex);
        glTexStorage2D(GL_TEXTURE_2D,
                       1, /* num levels */
                       GL_R8,
                       ctx->grey_width, ctx->grey_height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    start = get_time();
    glBindTexture(GL_TEXTURE_2D, ctx->cam_tex);

    for (int y = 0; y < 40; y++) {
        uint8_t *p = ((uint8_t *)ctx->grey_buffer_1_1.data()) + ctx->grey_width * y;
        memset(p, 0x80, ctx->grey_width / 2);
    }
    for (int y = 80; y < (ctx->grey_height / 2); y++) {
        uint8_t *p = ((uint8_t *)ctx->grey_buffer_1_1.data()) + ctx->grey_width * y;
        memset(p, 0x80, 40);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D,
                    0, /* level */
                    0, 0, /* x_off/y_off */
                    ctx->grey_width, ctx->grey_height,
                    GL_RED,
                    GL_UNSIGNED_BYTE, ctx->grey_buffer_1_1.data());

    end = get_time();
    duration_ns = end - start;

    LOGI("Uploaded top level luminance texture to GPU via glTexSubImage2D in %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    glUseProgram(scale_program_);
#endif

    start = get_time();

    glEnableVertexAttribArray(ctx->attrib_quad_rot_scale_pos);
    glVertexAttribPointer(ctx->attrib_quad_rot_scale_pos,
                          2, GL_FLOAT, GL_FALSE, sizeof(quad_strip[0]), (void *)0);
    glEnableVertexAttribArray(ctx->attrib_quad_rot_scale_tex_coords);
    glVertexAttribPointer(ctx->attrib_quad_rot_scale_tex_coords,
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

    //LOGI("Allocated pyramid level texture + fbo in %.3f%s",
    //     get_duration_ns_print_scale(duration_ns),
    //     get_duration_ns_print_scale_suffix(duration_ns));

    glViewport(0, 0, rotated_frame_width / 2, rotated_frame_height / 2);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisableVertexAttribArray(ctx->attrib_quad_rot_scale_pos);
    glDisableVertexAttribArray(ctx->attrib_quad_rot_scale_tex_coords);

    glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

    end = get_time();
    duration_ns = end - start;

    LOGI("Submitted level0 downsample in %.3f%s",
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

    LOGI("glGenerateMipmap took %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    glBindTexture(GL_TEXTURE_2D, 0);

    if (!ctx->read_back_fbo) {
        glGenFramebuffers(1, &ctx->read_back_fbo);
        glGenBuffers(1, &ctx->read_back_pbo);

        glBindFramebuffer(GL_FRAMEBUFFER, ctx->read_back_fbo);
        glBindTexture(GL_TEXTURE_2D, downsample_tex2d_);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               downsample_tex2d_, 1);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            LOGE("Famebuffer complete check failed");

        //LOGI("Allocated pyramid level texture + fbo in %.3f%s",
        //     get_duration_ns_print_scale(duration_ns),
        //     get_duration_ns_print_scale_suffix(duration_ns));

        glBindBuffer(GL_PIXEL_PACK_BUFFER, ctx->read_back_pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER,
                     (rotated_frame_width / 4) * (rotated_frame_height / 4),
                     nullptr, GL_DYNAMIC_READ);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, ctx->read_back_fbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, ctx->read_back_pbo);

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

    LOGI("glReadPixels took %.3f%s",
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

    LOGI("glMapBufferRange took %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));

    {
        dlib::timing::timer lv0_cpy_timer("Copied pyramid level0 frame for face detection from PBO in");

        tracking->face_detect_buf_width = rotated_frame_width / 4;
        tracking->face_detect_buf_height = rotated_frame_height / 4;

        /* TODO: avoid copying out of the PBO later (assuming we can get a
         * cached mapping)
         */
        LOGI("face detect scratch width = %d, height = %d",
             (int)tracking->face_detect_buf_width,
             (int)tracking->face_detect_buf_height);
        ctx->grey_face_detect_scratch.resize(tracking->face_detect_buf_width * tracking->face_detect_buf_height);
        memcpy(ctx->grey_face_detect_scratch.data(), pbo_ptr, ctx->grey_face_detect_scratch.size());

        tracking->face_detect_buf = ctx->grey_face_detect_scratch.data();
        LOGI("tracking->face_detect_buf = %p", tracking->face_detect_buf);
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    //glDeleteTextures(ctx->pyramid_tex_layers.size(), &ctx->pyramid_tex_layers[0]);
    //glDeleteFramebuffers(pyramid_fbos_.size(), &pyramid_fbos_[0]);

#endif /* DOWNSAMPLE_ON_GPU */

    ctx->need_new_scaled_frame = false;
    pthread_cond_signal(&ctx->scaled_frame_available_cond);

    pthread_mutex_unlock(&ctx->liveness_lock);
}

struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx)
{
    return &ctx->properties_state;
}

void
gm_context_set_event_callback(struct gm_context *ctx,
                              void (*event_callback)(struct gm_context *ctx,
                                                     struct gm_event *event,
                                                     void *user_data),
                              void *user_data)
{
    ctx->event_callback = event_callback;
    ctx->callback_data = user_data;
}

void
gm_context_enable(struct gm_context *ctx)
{
    request_frame(ctx);
}

/* Disable skeltal tracking */
void
gm_context_disable(struct gm_context *ctx)
{
}

void *
gm_frame_get_video_buffer(struct gm_frame *frame)
{
    return frame->video;
}

enum gm_format
gm_frame_get_video_format(struct gm_frame *frame)
{
    return frame->video_format;
}

