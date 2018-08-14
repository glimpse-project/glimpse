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
#include <glm/gtc/quaternion.hpp>
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
#include "random.hpp"

#include "xalloc.h"
#include "wrapper_image.h"
#include "infer_labels.h"
#include "infer_joints.h"
#include "rdt_tree.h"
#include "jip.h"
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

#define clampf(v, lo, hi) std::min(std::max(v, lo), hi)

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
using Random = effolkronium::random_thread_local;

#define DOWNSAMPLE_1_2
//#define DOWNSAMPLE_1_4

#ifdef DOWNSAMPLE_1_4

/* One implies the other... */
#ifndef DOWNSAMPLE_1_2
#define DOWNSAMPLE_1_2
#endif

#endif

#define TRACK_FRAMES 12

enum tracking_stage {
    TRACKING_STAGE_START,
    TRACKING_STAGE_GAP_FILLED,
    TRACKING_STAGE_DOWNSAMPLED,
    TRACKING_STAGE_GROUND_SPACE,

    TRACKING_STAGE_NAIVE_FLOOR,
    TRACKING_STAGE_NAIVE_CLUSTER,

    TRACKING_STAGE_CODEBOOK_SPACE,
    TRACKING_STAGE_CODEBOOK_CLASSIFY,
    TRACKING_STAGE_CODEBOOK_CLUSTER,

    TRACKING_STAGE_FILTER_CLUSTERS,
    TRACKING_STAGE_PROJECT_CLUSTERS,

    TRACKING_STAGE_SELECT_CANDIDATE_CLUSTER,
    TRACKING_STAGE_LABEL_INFERENCE,
    TRACKING_STAGE_JOINT_WEIGHTS,
    TRACKING_STAGE_JOINT_INFERENCE,
    TRACKING_STAGE_REFINE_SKELETON,

    TRACKING_STAGE_SANITIZE_SKELETON,

    TRACKING_STAGE_UPDATE_CODEBOOK,

    N_TRACKING_STAGES
};

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

// Depth pixel codewords for segmentation
struct seg_codeword
{
    float m;         // The mean value
    int n;           // The number of depth values in this codeword
    uint64_t ts;     // The frame timestamp this codeword was created on
    uint64_t tl;     // The last frame timestamp this codeword was used
    int nc;          // The number of times depth values consecutively fell
                     // into this codeword
};

// Depth pixel classification for segmentation
enum seg_class
{
    BG,       // Background
    FL,       // Flat
    FLK,      // Flickering
    FL_FLK,   // Flickering and flat
    TB,       // The bag (uninteresting foreground object)
    FG,       // Foreground
    CAN,      // Tracking candidate that didn't track
    TRK,      // Tracking
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

struct gm_bone
{
    float length;
    glm::quat angle;
    int head;
    int tail;
    bool length_corrected;
    bool angle_corrected;

    gm_bone() :
        length(0.f),
        head(-1),
        length_corrected(false),
        angle_corrected(false) {}
};

struct gm_skeleton {
    std::vector<struct gm_joint> joints;
    std::vector<struct gm_bone> bones;
    float confidence;
    float distance;
    uint64_t timestamp;

    gm_skeleton() :
      confidence(0.f),
      distance(0.f) {}
    gm_skeleton(int n_joints) :
      joints(n_joints),
      confidence(0.f),
      distance(0.f) {}
    gm_skeleton(const struct gm_skeleton *skeleton) :
      joints(skeleton->joints),
      bones(skeleton->bones),
      confidence(skeleton->confidence),
      distance(skeleton->distance),
      timestamp(skeleton->timestamp) {}
};

struct image_generator {
    const char *name;
    const char *desc;
    bool (*create_rgb_image)(struct gm_tracking *tracking,
                             int *width,
                             int *height,
                             uint8_t **output);
};

struct gm_pipeline_stage {
    enum tracking_stage stage_id;
    const char *name;
    const char *desc;

    uint64_t total_time_ns;
    int n_invocations;

    std::vector<image_generator> images;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
};

/* A pipeline_stage maintains global information about a stage while
 * _stage_data can track information about a specific tracking run
 */
struct gm_pipeline_stage_data {
    uint64_t duration_ns;

    //std::vector<struct gm_point_rgba> debug_point_cloud;
    //std::vector<struct gm_point_rgba> debug_lines;
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

    /* Note: these are derived from the corresponding frame intrinsics except
     * they take into account the device rotation at the start of tracking.
     */
    struct gm_intrinsics depth_camera_intrinsics;
    struct gm_intrinsics video_camera_intrinsics;

    /* This is currently just a copy of ctx->training_camera_intrinsics */
    struct gm_intrinsics training_camera_intrinsics;

    /* XXX: these are currently a copy of ctx->basis_depth_to_video_extrinsics
     * and don't take into account device rotation
     */
    struct gm_extrinsics depth_to_video_extrinsics;
    bool extrinsics_set;

    struct gm_frame *frame;

    // Depth data, in meters
    float *depth;

    // Label inference data
    uint8_t *label_map;

    // Label probability tables
    float *label_probs;

    // The unprojected full-resolution depth cloud
    pcl::PointCloud<pcl::PointXYZL>::Ptr depth_cloud;

    // The ground-aligned segmentation-resolution depth cloud
    pcl::PointCloud<pcl::PointXYZL>::Ptr ground_cloud;

    // Labels based on depth value classification
    pcl::PointCloud<pcl::PointXYZL>::Ptr depth_class;

    // Labels based on clustering after plane removal
    pcl::PointCloud<pcl::Label>::Ptr cluster_labels;

    std::vector<struct gm_point_rgba> debug_cloud;

    // While building the debug_cloud we sometimes track indices that map
    // back to some other internal cloud which may get used for colouring
    // the debug points.
    std::vector<int> debug_cloud_indices;

    std::vector<struct gm_point_rgba> debug_lines;

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

    std::vector<gm_pipeline_stage_data> stage_data;

    uint64_t duration_ns;
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

    int max_depth_pixels;
    int max_video_pixels;

    struct gm_intrinsics training_camera_intrinsics;

    /* '_basis' here implies that the transform does not take into account how
     * video/depth data may be rotated to match the device orientation
     *
     * FIXME: don't just copy this to the tracking state without considering
     * the device orientation.
     */
    struct gm_extrinsics basis_depth_to_video_extrinsics;
    bool basis_extrinsics_set;

    /* These stages represent the processing pipeline for skeletal tracking
     * and let us group statistics and UI properties
     */
    std::vector<struct gm_pipeline_stage> stages;

    bool reset_codebook;
    struct gm_pose codebook_pose;
    glm::mat4 start_to_codebook;
    std::vector<std::vector<struct seg_codeword>> depth_seg;
    std::vector<struct seg_codeword *> depth_seg_bg;

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

    int debug_pipeline_stage;
    int debug_cloud_mode;

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

    bool apply_depth_distortion;

    float min_depth;
    float max_depth;
    int seg_res;

    float floor_threshold;
    float cluster_tolerance;

    float cluster_min_width;
    float cluster_min_height;
    float cluster_min_depth;

    float cluster_max_width;
    float cluster_max_height;
    float cluster_max_depth;

    bool motion_detection;
    float seg_tb;
    float seg_tf;
    int seg_N;
    int seg_b;
    int seg_gamma;
    int seg_alpha;
    float seg_psi;
    float seg_timeout;

    bool flip_labels;

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

    bool skeleton_validation;
    float skeleton_min_confidence;
    float skeleton_max_distance;

    int n_depth_color_stops;
    float depth_color_stops_range;
    struct color_stop *depth_color_stops;

    int n_heat_color_stops;
    float heat_color_stops_range;
    struct color_stop *heat_color_stops;

    std::vector<struct gm_ui_enumerant> cloud_stage_enumerants;
    std::vector<struct gm_ui_enumerant> cloud_mode_enumerants;
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

static void
tracking_draw_line(struct gm_tracking_impl *tracking,
                   float x0, float y0, float z0,
                   float x1, float y1, float z1,
                   uint32_t rgba)
{
    struct gm_point_rgba p0 = { x0, -y0, z0, rgba };
    struct gm_point_rgba p1 = { x1, -y1, z1, rgba };
    tracking->debug_lines.push_back(p0);
    tracking->debug_lines.push_back(p1);
}

static inline float
distance_between(const float *point1, const float *point2)
{
    return sqrtf(powf(point1[0] - point2[0], 2.f) +
                 powf(point1[1] - point2[1], 2.f) +
                 powf(point1[2] - point2[2], 2.f));
}

static inline bool
is_bone_length_diff(const struct gm_bone &ref_bone,
                    const struct gm_bone &bone,
                    float max_variance)
{
    return fabsf(bone.length - ref_bone.length) > max_variance;
}

static inline float
bone_angle_diff(const struct gm_bone *bone,
                const struct gm_skeleton *ref_skel,
                const struct gm_skeleton *skel)
{
    glm::vec3 bone_vec = glm::vec3(skel->joints[bone->tail].x -
                                   skel->joints[bone->head].x,
                                   skel->joints[bone->tail].y -
                                   skel->joints[bone->head].y,
                                   skel->joints[bone->tail].z -
                                   skel->joints[bone->head].z);
    glm::vec3 ref_vec = glm::vec3(ref_skel->joints[bone->tail].x -
                                  ref_skel->joints[bone->head].x,
                                  ref_skel->joints[bone->tail].y -
                                  ref_skel->joints[bone->head].y,
                                  ref_skel->joints[bone->tail].z -
                                  ref_skel->joints[bone->head].z);
    float angle = glm::degrees(acosf(
        glm::dot(glm::normalize(bone_vec), glm::normalize(ref_vec))));
    while (angle > 180.f) angle -= 360.f;
    return angle;
}

static inline bool
is_bone_angle_diff(const struct gm_bone &bone,
                   const struct gm_skeleton &ref_skel,
                   const struct gm_skeleton &skel,
                   float max_angle)
{
    float angle = bone_angle_diff(&bone, &ref_skel, &skel);
    float time = skel.timestamp > ref_skel.timestamp ?
        (float)((skel.timestamp - ref_skel.timestamp) / 1e9) :
        (float)((ref_skel.timestamp - skel.timestamp) / 1e9);
    float angle_delta = fabsf(angle) / time;

    return angle_delta > max_angle;
}

static int
is_skeleton_diff(const struct gm_context *ctx,
                 const struct gm_skeleton &skel,
                 const struct gm_skeleton &ref)
{
    int violations = 0;
    for (unsigned i = 0; i < ref.bones.size(); ++i) {
        const struct gm_bone &ref_bone = ref.bones[i];
        if (ref_bone.head < 0) {
            continue;
        }

        bool bone_found = false;
        for (unsigned j = 0; j < skel.bones.size(); ++j) {
            const struct gm_bone &bone = skel.bones[i];

            if (bone.head != ref_bone.head ||
                bone.tail != ref_bone.tail) {
                continue;
            }

            bone_found = true;

            if (is_bone_length_diff(bone, ref_bone,
                                    ctx->bone_length_variance)) {
                ++violations;
            }
            if (is_bone_angle_diff(bone, ref, skel,
                                   ctx->bone_rotation_variance)) {
                ++violations;
            }
            break;
        }

        if (!bone_found) {
            violations += 2;
        }
    }

    return violations;
}

static bool
compare_skeletons(const struct gm_skeleton &skel1,
                  const struct gm_skeleton &skel2)
{
    return (skel1.distance == skel2.distance) ?
        skel1.confidence > skel2.confidence :
        skel1.distance < skel2.distance;
}

static const struct gm_bone *
find_bone(const std::vector<struct gm_bone> &bones, int head, int tail)
{
    for (std::vector<struct gm_bone>::const_iterator it = bones.begin();
         it != bones.end(); ++it) {
        const struct gm_bone &candidate = *it;
        if ((head == -1 || candidate.head == head) &&
            (tail == -1 || candidate.tail == tail)) {
            return &candidate;
        }
    }

    return NULL;
}

const struct gm_bone *
gm_skeleton_find_bone(const struct gm_skeleton *skeleton, int head, int tail)
{
    return find_bone(skeleton->bones, head, tail);
}

static void
update_bones(struct gm_context *ctx,
             struct gm_skeleton &skeleton)
{
    for (std::vector<struct gm_bone>::iterator it = skeleton.bones.begin();
         it != skeleton.bones.end(); ++it) {
        struct gm_bone &bone = *it;

        // Calculate the bone length
        struct gm_joint &head = skeleton.joints[bone.head];
        struct gm_joint &tail = skeleton.joints[bone.tail];
        float dist = distance_between(&tail.x, &head.x);
        bone.length = dist;

        // Find the bone that leads to this one to determine the
        // angle between them.
        const struct gm_bone *parent = find_bone(skeleton.bones, -1, bone.head);
        if (parent) {
            // Calculate the angle between this bone and its parent
            struct gm_joint &last_head = skeleton.joints[parent->head];
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

            bone.angle =
                glm::angleAxis(angle, axis);

#if 0
            {
                // Debugging
                glm::vec3 tb = (parent_vec * glm::mat3_cast(bone.angle)) * dist;
                gm_debug(ctx->log, "XXX tail: %.2f, %.2f, %.2f, "
                         "transformed parent: %.2f, %.2f, %.2f",
                         tail.x, tail.y, tail.z,
                         tb.x + head.x, tb.y + head.y, tb.z + head.z);
            }
#endif
        } else {
            bone.angle = glm::quat();
        }
    }
}

static void
build_bones(struct gm_context *ctx,
            struct gm_skeleton &skeleton,
            int joint_no = 0,
            int last_joint_no = -1)
{
    if (joint_no != last_joint_no) {
        if (skeleton.joints[joint_no].confidence > 0.f &&
            last_joint_no != -1 &&
            skeleton.joints[last_joint_no].confidence > 0.f)
        {
            struct gm_bone bone;
            bone.length = 0;
            bone.head = last_joint_no;
            bone.tail = joint_no;
            bone.angle = glm::quat();
            bone.length_corrected = false;
            bone.angle_corrected = false;

            skeleton.bones.push_back(bone);
        }
    }

    for (int i = 0; i < ctx->joint_stats[joint_no].n_connections; ++i) {
        if (ctx->joint_stats[joint_no].connections[i] == last_joint_no) {
            continue;
        }
        build_bones(ctx, skeleton,
                    ctx->joint_stats[joint_no].connections[i], joint_no);
    }

    if (joint_no == 0 && last_joint_no == -1) {
        update_bones(ctx, skeleton);
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
            skeleton.joints[joint_no].name = joint_name(joint_no);
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
            skeleton.joints[joint_no].name = NULL;
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

    // If we tracked the previous frame, predict a skeleton to use as a
    // reference for refinement.
    int skel_diff = 0;
    if (ctx->n_tracking && ctx->tracking_history[0]) {
        skel_diff = is_skeleton_diff(ctx, skeleton,
                                     ctx->tracking_history[0]->skeleton);
    }

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
            candidate_skeleton.timestamp = skeleton.timestamp;

            Joint *joint = (Joint *)l->data;
            candidate_skeleton.joints[j].name = joint_name(j);
            candidate_skeleton.joints[j].x = joint->x;
            candidate_skeleton.joints[j].y = joint->y;
            candidate_skeleton.joints[j].z = joint->z;
            candidate_skeleton.joints[j].confidence = joint->confidence;
            candidate_skeleton.joints[j].predicted = false;
            candidate_skeleton.confidence += joint->confidence;

            build_skeleton(ctx, result, candidate_skeleton, j, j);
            build_bones(ctx, candidate_skeleton);

            int cand_diff = 0;
            if (ctx->n_tracking && ctx->tracking_history[0]) {
                cand_diff =
                    is_skeleton_diff(ctx, candidate_skeleton,
                                     ctx->tracking_history[0]->skeleton);
            }
            if (cand_diff < skel_diff &&
                compare_skeletons(candidate_skeleton, skeleton)) {
                std::swap(skeleton, candidate_skeleton);
                skel_diff = cand_diff;
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
                         distance, distance / prev_dist,
                         parent_joint.confidence);
                    parent_joint.x = prediction.x;
                    parent_joint.y = prediction.y;
                    parent_joint.z = prediction.z;
                    parent_joint.predicted = true;
                }
            }
        }
    }

    // Update the bone metadata
    update_bones(ctx, skeleton);

    // For each bone, we compare the bone length and look at the change in
    // angular acceleration. Bone length shouldn't change of course, but
    // tracking isn't perfect, so we allow some squishiness. If either exceed
    // the set thresholds, we replace this bone with a prediction based on
    // previous confident bones.
    for (std::vector<struct gm_bone>::iterator it = skeleton.bones.begin();
         it != skeleton.bones.end(); ++it) {
        struct gm_bone &bone = *it;

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
        const struct gm_bone *prev_length = NULL;
        struct gm_skeleton *prev_angle = NULL;
        for (int i = 0; i < ctx->n_tracking; ++i) {
            const struct gm_bone *prev_bone =
                find_bone(ctx->tracking_history[i]->skeleton.bones,
                          bone.head, bone.tail);
            if (!prev_bone) {
                continue;
            }
            if (!prev_bone->length_corrected && !prev_length) {
                prev_length = prev_bone;
            }
            if (!prev_bone->angle_corrected && !prev_angle) {
                prev_angle = &ctx->tracking_history[i]->skeleton;
            }
        }

        if (prev_length) {
            // If the bone length has changed significantly, adjust the length
            // so that it remains within our acceptable bounds of variation.
            if (is_bone_length_diff(bone, *prev_length,
                                    ctx->bone_length_variance)) {
                float new_length = (bone.length > prev_length->length) ?
                    prev_length->length + ctx->bone_length_variance :
                    prev_length->length - ctx->bone_length_variance;

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

                // Refresh bone info now the joint has changed
                update_bones(ctx, skeleton);

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
        if (!find_bone(skeleton.bones, -1, bone.head)) {
            continue;
        }

        if (prev_angle) {
            // If the bone angle has changed quicker than we expected, use the
            // previous bone angle.
            if (is_bone_angle_diff(bone, skeleton, *prev_angle,
                                   ctx->bone_rotation_variance)) {
#if 0
                // It feels like mixing in some of the angle from the spurious
                // tracking might be nice, but generally, if we're doing this
                // correction it was because the tracked point is nonsense.
                glm::mat3 rotate = glm::mat3_cast(
                    glm::slerp(prev_bone.angle, bone.angle,
                               (ctx->bone_rotation_variance / angle_delta)));
#endif
                // We don't care about using a corrected bone for this, we want
                // the angle to change smoothly regardless of whether it's
                // based on a corrected value or not.
                const struct gm_bone *abs_prev_bone =
                    find_bone(ctx->tracking_history[0]->skeleton.bones,
                              bone.head, bone.tail);
                const struct gm_bone *parent_bone =
                    find_bone(skeleton.bones, -1, bone.head);
                if (abs_prev_bone && parent_bone) {
                    glm::mat3 rotate = glm::mat3_cast(abs_prev_bone->angle);

                    glm::vec3 parent_vec = glm::normalize(
                        glm::vec3(skeleton.joints[parent_bone->tail].x -
                                  skeleton.joints[parent_bone->head].x,
                                  skeleton.joints[parent_bone->tail].y -
                                  skeleton.joints[parent_bone->head].y,
                                  skeleton.joints[parent_bone->tail].z -
                                  skeleton.joints[parent_bone->head].z));

                    glm::vec3 new_tail = ((parent_vec * rotate) * bone.length);
                    new_tail.x += skeleton.joints[bone.head].x;
                    new_tail.y += skeleton.joints[bone.head].y;
                    new_tail.z += skeleton.joints[bone.head].z;

                    skeleton.joints[bone.tail].x = new_tail.x;
                    skeleton.joints[bone.tail].y = new_tail.y;
                    skeleton.joints[bone.tail].z = new_tail.z;
                    skeleton.joints[bone.tail].predicted = true;
                    bone.angle_corrected = true;

                    // Refresh bone info now the joint has changed
                    update_bones(ctx, skeleton);

                    gm_debug(ctx->log, "Bone angle correction (%s->%s) "
                             "(%.2f, %.2f, %.2f)->(%.2f, %.2f, %.2f)",
                             joint_name(bone.head), joint_name(bone.tail),
                             skeleton.joints[bone.head].x,
                             skeleton.joints[bone.head].y,
                             skeleton.joints[bone.head].z,
                             skeleton.joints[bone.tail].x,
                             skeleton.joints[bone.tail].y,
                             skeleton.joints[bone.tail].z);
                }
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

template<typename PointT>
class LabelComparator: public pcl::Comparator<PointT>
{
  public:
    typedef typename pcl::Comparator<PointT>::PointCloud PointCloud;
    typedef typename pcl::Comparator<PointT>::PointCloudConstPtr
        PointCloudConstPtr;

    typedef boost::shared_ptr<LabelComparator<PointT>> Ptr;
    typedef boost::shared_ptr<const LabelComparator<PointT>> ConstPtr;

    using pcl::Comparator<PointT>::input_;

    LabelComparator()
      : depth_threshold_(0.03f) {
    }

    virtual
    ~LabelComparator() {
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
        if ((input_->points[idx1].label == FLK ||
             input_->points[idx1].label == FG) &&
            (input_->points[idx2].label == FLK ||
             input_->points[idx2].label == FG)) {
            return fabsf(input_->points[idx1].z - input_->points[idx2].z) <
                depth_threshold_;
        }

        return false;
    }

  protected:
    float depth_threshold_;
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

/* TODO: combine the to_start and to_codebook matrices
 */
static int
project_point_into_codebook(pcl::PointXYZL *point,
                            glm::mat4 to_start,
                            glm::mat4 start_to_codebook,
                            struct gm_intrinsics *intrinsics,
                            int seg_res)
{
    if (std::isnan(point->z))
        return -1;

    const int width = intrinsics->width / seg_res;
    const int height = intrinsics->height / seg_res;
    const float fx = intrinsics->fx;
    const float fy = intrinsics->fy;
    const float cx = intrinsics->cx;
    const float cy = intrinsics->cy;

    glm::vec4 pt(point->x, point->y, point->z, 1.f);
    pt = (to_start * pt);
    pt = (start_to_codebook * pt);
    point->x = pt.x;
    point->y = pt.y;
    point->z = pt.z;

    float nx = ((pt.x * fx / pt.z) + cx);
    float ny = ((pt.y * fy / pt.z) + cy);

    /* XXX: check the rounding here... */
    int dnx = (int)roundf(nx / seg_res);
    int dny = (int)roundf(ny / seg_res);

    if (dnx < 0 || dnx >= width ||
        dny < 0 || dny >= height)
    {
        return -1;
    }

    return width * dny + dnx;
}

static void
update_depth_codebook(struct gm_context *ctx,
                      struct gm_tracking_impl *tracking,
                      glm::mat4 to_start,
                      glm::mat4 to_codebook,
                      int seg_res)
{
    uint64_t start = get_time();

    int n_codewords = 0;
    struct gm_intrinsics intrinsics = tracking->depth_camera_intrinsics;

    unsigned depth_class_size = tracking->depth_class->points.size();
    for (unsigned depth_off = 0; depth_off < depth_class_size; ++depth_off) {
        pcl::PointXYZL point = tracking->depth_class->points[depth_off];

        if (std::isnan(point.z)) {
            continue;
        } else {
            int off = project_point_into_codebook(&point,
                                                  to_start,
                                                  to_codebook,
                                                  &intrinsics,
                                                  seg_res);
            // Falls outside of codebook so we can't classify...
            if (off < 0)
                continue;

            // At this point z has been projected into the coordinate space of
            // the codebook
            float depth = point.z;

            // Look to see if this pixel falls into an existing codeword
            struct seg_codeword *codeword = NULL;
            std::vector<struct seg_codeword> &codewords = ctx->depth_seg[off];
            std::vector<struct seg_codeword>::iterator it;
            for (it = codewords.begin(); it != codewords.end(); ++it) {
                struct seg_codeword &candidate = *it;

                if (fabsf(depth - candidate.m) < ctx->seg_tb) {
                    codeword = &candidate;
                    break;
                }
            }

            // Don't update pixels that we're tracking
            if (tracking->depth_class->points[depth_off].label == TRK) {
                if (codeword) {
                    if (codewords.size() > 2) {
                        std::swap(*it, codewords.back());
                        codewords.pop_back();
                    } else {
                        codewords.erase(it);
                    }
                }
                continue;
            }

            const uint64_t t = tracking->frame->timestamp;

            // Create a new codeword if one didn't fit
            if (!codeword) {
                codewords.push_back({ 0, 0, t, t, 0 });
                codeword = &codewords.back();
            }

            // Update the codeword info
            // Update the mean depth
            float n = (float)std::min(ctx->seg_N, codeword->n);
            codeword->m = ((n * codeword->m) + depth) / (n + 1.f);

            // Increment number of depth values
            ++codeword->n;

            // Increment consecutive number of depth values if its happened in
            // consecutive frames
            if (!ctx->n_tracking ||
                codeword->tl != ctx->latest_tracking->frame->timestamp) {
                ++codeword->nc;
            }

            // Track the latest timestamp to touch this codeword
            codeword->tl = t;

            // Keep track of the amount of codewords we have
            n_codewords += (int)codewords.size();
        }
    }

    uint64_t end = get_time();
    uint64_t duration = end - start;
    LOGI("Codeword update (%.2f codewords/pix) took %.3f%s",
         n_codewords / (float)(tracking->depth_class->width *
                               tracking->depth_class->height),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));
}

static glm::mat4
pose_to_matrix(struct gm_pose &pose)
{
    glm::quat rot_start_to_dev(
        pose.orientation[3],
        pose.orientation[0],
        pose.orientation[1],
        pose.orientation[2]);

    glm::vec3 mov_start_to_dev(
        pose.translation[0],
        pose.translation[1],
        pose.translation[2]);

    return glm::mat4_cast(rot_start_to_dev) *
        glm::translate(glm::mat4(1.f), mov_start_to_dev);
}

static inline bool
gm_compare_depth(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                 int x1, int y1, int x2, int y2,
                 float tolerance)
{
    float d1 = cloud->points[y1 * cloud->width + x1].z;
    float d2 = cloud->points[y2 * cloud->width + x2].z;
    if (std::isnan(d1) || std::isnan(d2)) {
        return false;
    }

    /* We assume that there's nothing between the camera and the person
     * we're trying to segment/detect and considering that their can be
     * quite large jumps in depth when arms cross the body we have no
     * threshold moving towards the camera...
     *
     * NB: d2 corresponds to a point that we've already decided is inside
     * the cluster/body
     */
    if (d1 < d2)
        return true;

    return fabsf(d1 - d2) <= tolerance;
}

/* simple hash to cheaply randomize how we will gaps in our data without
 * too much bias
 */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void
pcl_xyzl_cloud_from_buf_with_fill_and_threshold(struct gm_context *ctx,
                                                struct gm_tracking_impl *tracking,
                                                pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud,
                                                float *depth,
                                                struct gm_intrinsics *intrinsics)
{
    float nan = std::numeric_limits<float>::quiet_NaN();

    int width = intrinsics->width;
    int height = intrinsics->height;

    // We gap fill with a 3x3 box filter and special case the borders...
    int x_end = width - 1;
    int y_end = height - 1;

    float fx = intrinsics->fx;
    float fy = intrinsics->fy;
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    float cx = intrinsics->cx;
    float cy = intrinsics->cy;

    pcl_cloud->width = width;
    pcl_cloud->height = height;
    pcl_cloud->points.resize(width * height);
    pcl_cloud->is_dense = false;

    float z_min = ctx->min_depth;
    float z_max = ctx->max_depth;

#define copy_row(Y) do { \
    int y = Y; \
    int row = y * width; \
    for (int x = 0; x < width; x++) { \
        pcl::PointXYZL point; \
        point.z = depth[row + x]; \
        \
        if (!std::isnormal(point.z) || \
            point.z < z_min || \
            point.z > z_max) \
        { \
            point.x = point.y = point.z = nan; \
            point.label = -1; \
            pcl_cloud->points[row + x] = point; \
            continue; \
        } \
        point.x = (x - cx) * point.z * inv_fx; \
        point.y = ((y - cy) * point.z * inv_fy); \
        point.label = -1; \
        pcl_cloud->points[row + x] = point; \
    } \
} while(0)

    copy_row(0);

    uint32_t seed = 1;
    for (int y = 1; y < y_end; y++) {
        for (int x = 0; x < width; x++) {
            int off = y * width + x;
            pcl::PointXYZL point;
            if (x == 0 || x == x_end) {
                // Just copy the left/right border
                point.z = depth[off];
            } else {
                int y_up = y - 1;
                int y_down = y - 1;
                float neighbours[8] = {
                    depth[y_up * width + (x-1)],
                    depth[y_up * width + x],
                    depth[y_up * width + (x+1)],
                    depth[y * width + (x-1)],
                    depth[y * width + (x+1)],
                    depth[y_down * width + (x-1)],
                    depth[y_down * width + x],
                    depth[y_down * width + (x+1)],
                };

                uint32_t rnd = xorshift32(&seed);
                //printf("XOR RND (idx=%d): |%*s'%*s|\n",
                //       rnd, (rnd%8), (rnd%8), "", 7-(rnd%8), "");
                point.z = neighbours[rnd % 8];
                for (int i = 1; !std::isnormal(point.z) && i < 8; i++) {
                    point.z = neighbours[(rnd + i) % 8];
                }
            }

            if (!std::isnormal(point.z) ||
                point.z < z_min ||
                point.z > z_max)
            {
                point.x = point.y = point.z = nan;
                point.label = -1;
                pcl_cloud->points[off] = point;
                continue;
            }

            point.x = (x - cx) * point.z * inv_fx;
            point.y = ((y - cy) * point.z * inv_fy);
            point.label = -1;
            pcl_cloud->points[off] = point;
        }
    }

    copy_row(height - 1);
#undef copy_row
}

static void
add_debug_cloud_xyz_from_pcl_xyzl(struct gm_context *ctx,
                                  struct gm_tracking_impl *tracking,
                                  pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;
    debug_cloud.resize(debug_cloud.size() + pcl_cloud->size());
    debug_cloud_indices.resize(debug_cloud_indices.size() + pcl_cloud->size());

    for (unsigned i = 0; i < pcl_cloud->size(); i++) {
        debug_cloud[i].x = pcl_cloud->points[i].x;
        debug_cloud[i].y = -pcl_cloud->points[i].y; // FIXME
        debug_cloud[i].z = pcl_cloud->points[i].z;
        debug_cloud[i].rgba = 0xffffffff;
        debug_cloud_indices[i] = i;
    }
}

static void
add_debug_cloud_xyz_from_pcl_xyzl_transformed(struct gm_context *ctx,
                                              struct gm_tracking_impl *tracking,
                                              pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud,
                                              glm::mat4 transform)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;
    debug_cloud.resize(debug_cloud.size() + pcl_cloud->size());
    debug_cloud_indices.resize(debug_cloud_indices.size() + pcl_cloud->size());

    for (unsigned i = 0; i < pcl_cloud->size(); i++) {
        glm::vec4 pt(pcl_cloud->points[i].x,
                     pcl_cloud->points[i].y,
                     pcl_cloud->points[i].z,
                     1.f);
        pt = (transform * pt);

        debug_cloud[i].x = pt.x;
        debug_cloud[i].y = - pt.y; // FIXME
        debug_cloud[i].z = pt.z;
        debug_cloud[i].rgba = 0xffffffff;
        debug_cloud_indices[i] = i;
    }
}

static void
add_debug_cloud_xyz_from_pcl_xyzl_and_indices(struct gm_context *ctx,
                                              struct gm_tracking_impl *tracking,
                                              pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud,
                                              std::vector<int> &indices)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;
    debug_cloud.resize(debug_cloud.size() + indices.size());
    debug_cloud_indices.resize(debug_cloud.size() + indices.size());

    for (unsigned i = 0; i < indices.size(); i++) {
        debug_cloud[i].x = pcl_cloud->points[indices[i]].x;
        debug_cloud[i].y = -pcl_cloud->points[indices[i]].y; // FIXME
        debug_cloud[i].z = pcl_cloud->points[indices[i]].z;
        debug_cloud[i].rgba = 0xffffffff;
        debug_cloud_indices[i] = indices[i];
    }
}

/* 'Dense' here means we don't expect any NAN values */
static void
add_debug_cloud_xyz_from_dense_depth_buf(struct gm_context *ctx,
                                         struct gm_tracking_impl *tracking,
                                         float *depth,
                                         struct gm_intrinsics *intrinsics)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    int width = intrinsics->width;
    int height = intrinsics->height;

    const float fx = intrinsics->fx;
    const float fy = intrinsics->fy;
    const float inv_fx = 1.0f / fx;
    const float inv_fy = 1.0f / fy;
    const float cx = intrinsics->cx;
    const float cy = intrinsics->cy;

    foreach_xy_off(width, height) {
        struct gm_point_rgba point;

        point.z = depth[off];

        if (!std::isnormal(point.z))
            continue;

        point.x = (x - cx) * point.z * inv_fx;
        point.y = -((y - cy) * point.z * inv_fy); // FIXME

        debug_cloud.push_back(point);
    }
}

static void
add_debug_cloud_xyz_of_codebook_space(struct gm_context *ctx,
                                      struct gm_tracking_impl *tracking,
                                      pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud,
                                      glm::mat4 to_start,
                                      glm::mat4 start_to_codebook,
                                      struct gm_intrinsics *intrinsics,
                                      int seg_res)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    for (unsigned i = 0; i < pcl_cloud->size(); i++) {
        pcl::PointXYZL pcl_point = pcl_cloud->points[i];
        struct gm_point_rgba point;

        project_point_into_codebook(&pcl_point,
                                    to_start,
                                    start_to_codebook,
                                    &tracking->depth_camera_intrinsics,
                                    seg_res);
        point.x = pcl_point.x;
        point.y = -pcl_point.y; // FIXME
        point.z = pcl_point.z;
        point.rgba = 0xffffffff;

        debug_cloud.push_back(point);
        debug_cloud_indices.push_back(i);
    }
}

static void
depth_classification_to_rgb(enum seg_class label, uint8_t *rgb_out)
{
    switch(label) {
    case BG:
        rgb_out[0] = 0x00;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0x00;
        break;
    case FL:
        rgb_out[0] = 0xC0;
        rgb_out[1] = 0xC0;
        rgb_out[2] = 0xC0;
        break;
    case FLK:
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0x00;
        break;
    case FL_FLK:
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xA0;
        rgb_out[2] = 0x00;
        break;
    case TB:
        rgb_out[0] = 0x00;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0xFF;
        break;
    case FG:
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0xFF;
        break;
    case CAN:
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0x00;
        break;
    case TRK:
        rgb_out[0] = 0x00;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0x00;
        break;
    default:
        // Invalid/unhandled value
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0xFF;
        break;
    }
}

/* Used to colour the debug cloud for most pipeline stages... */
static void
colour_debug_cloud(struct gm_context *ctx,
                   struct gm_tracking_impl *tracking,
                   pcl::PointCloud<pcl::PointXYZL>::Ptr indexed_pcl_cloud,
                   bool classified)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &indices = tracking->debug_cloud_indices;

    if (ctx->debug_cloud_mode == 1) {
        const float vid_fx = tracking->video_camera_intrinsics.fx;
        const float vid_fy = tracking->video_camera_intrinsics.fy;
        const float vid_cx = tracking->video_camera_intrinsics.cx;
        const float vid_cy = tracking->video_camera_intrinsics.cy;

        int vid_width = 0;
        int vid_height = 0;
        uint8_t *vid_rgb = NULL;
        gm_tracking_create_rgb_video(&tracking->base, &vid_width, &vid_height, &vid_rgb);
        if (vid_rgb) {
            if (indexed_pcl_cloud && indices.size()) {
                for (unsigned i = 0; i < indices.size(); i++) {
                    float x = indexed_pcl_cloud->points[indices[i]].x;
                    float y = indexed_pcl_cloud->points[indices[i]].y;
                    float z = indexed_pcl_cloud->points[indices[i]].z;

                    if (!std::isnormal(z))
                        continue;

                    // Reproject the depth coordinates into video space
                    // TODO: Support extrinsics
                    int vx = clampf(x * vid_fx / z + vid_cx, 0.0f, (float)vid_width - 1);
                    int vy = clampf(y * vid_fy / z + vid_cy, 0.0f, (float)vid_height - 1);
                    int v_off = vy * vid_width * 3 + vx * 3;

                    debug_cloud[i].rgba = (((uint32_t)vid_rgb[v_off])<<24 |
                                           ((uint32_t)vid_rgb[v_off+1])<<16 |
                                           ((uint32_t)vid_rgb[v_off+2])<<8 |
                                           0xff);
                }
            } else {
                for (unsigned off = 0; off < debug_cloud.size(); off++) {
                    float x = debug_cloud[off].x;
                    float y = -debug_cloud[off].y; // FIXME
                    float z = debug_cloud[off].z;

                    if (!std::isnormal(z))
                        continue;

                    // Reproject the depth coordinates into video space
                    // TODO: Support extrinsics
                    int vx = clampf(x * vid_fx / z + vid_cx, 0.0f, (float)vid_width - 1);
                    int vy = clampf(y * vid_fy / z + vid_cy, 0.0f, (float)vid_height - 1);
                    int v_off = vy * vid_width * 3 + vx * 3;

                    debug_cloud[off].rgba = (((uint32_t)vid_rgb[v_off])<<24 |
                                             ((uint32_t)vid_rgb[v_off+1])<<16 |
                                             ((uint32_t)vid_rgb[v_off+2])<<8 |
                                             0xff);
                }
            }

            free(vid_rgb);
        }
    } else if (ctx->debug_cloud_mode == 2) {
        for (unsigned off = 0; off < debug_cloud.size(); off++) {
            float z = debug_cloud[off].z;

            if (!std::isnormal(z))
                continue;

            struct color rgb = stops_color_from_val(ctx->depth_color_stops,
                                                    ctx->n_depth_color_stops,
                                                    ctx->depth_color_stops_range,
                                                    z);
            debug_cloud[off].rgba = (((uint32_t)rgb.r)<<24 |
                                     ((uint32_t)rgb.g)<<16 |
                                     ((uint32_t)rgb.b)<<8 |
                                     0xff);
        }
    } else if (ctx->debug_cloud_mode == 3 && classified) {
        if (indices.size()) {
            for (unsigned i = 0; i < indices.size(); i++) {
                enum seg_class label =
                    (enum seg_class)indexed_pcl_cloud->points[indices[i]].label;
                uint8_t rgb[3];
                depth_classification_to_rgb(label, rgb);
                debug_cloud[i].rgba = (((uint32_t)rgb[0])<<24 |
                                       ((uint32_t)rgb[1])<<16 |
                                       ((uint32_t)rgb[2])<<8 |
                                       0xff);
            }
        }
    }
}


struct PointCmp {
    int x;
    int y;
    int lx;
    int ly;
};

struct pipeline_scratch_state
{
    glm::mat4 to_start;
    glm::mat4 to_ground;

    // naive or codebook segmentation/clustering
    std::vector<pcl::PointIndices> cluster_indices;

    // naive segmentation
    int naive_fx;
    int naive_fy;
    std::vector<bool> done_mask;
    std::queue<struct PointCmp> flood_fill;
    float naive_floor_y;

    // cluster_select
    std::vector<pcl::PointIndices> persons;

    // cluster_project
    std::vector<float*> depth_images;

    // per-cluster inference
    float *depth_image;
    float *weights;
    float *label_probs;
    InferredJoints *joints_candidate;
    struct gm_skeleton candidate_skeleton;

    int best_person;

    pipeline_scratch_state(int n_joints) :
      candidate_skeleton(n_joints) {}
};

static void
stage_gap_fill_cb(struct gm_tracking_impl *tracking,
                  struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    if (!tracking->depth_cloud) {
        tracking->depth_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(
            new pcl::PointCloud<pcl::PointXYZL>);
    }

    pcl_xyzl_cloud_from_buf_with_fill_and_threshold(ctx,
                                                    tracking,
                                                    tracking->depth_cloud,
                                                    tracking->depth,
                                                    &tracking->
                                                     depth_camera_intrinsics);
}

static void
stage_gap_fill_debug_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_cloud);
    colour_debug_cloud(ctx, tracking, tracking->depth_cloud, false);
}

static void
stage_downsample_cb(struct gm_tracking_impl *tracking,
                    struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    // Person detection can happen in a sparser cloud made from a downscaled
    // version of the depth buffer. This is significantly cheaper than using a
    // voxel grid, which would produce better results but take a lot longer
    // doing so and give us less useful data structures.
    int seg_res = ctx->seg_res;
    if (seg_res == 1) {
        tracking->depth_class = tracking->depth_cloud;
    } else {
        if (!tracking->depth_class ||
            tracking->depth_class == tracking->depth_cloud) {
            tracking->depth_class = pcl::PointCloud<pcl::PointXYZL>::Ptr(
                new pcl::PointCloud<pcl::PointXYZL>);
        }

        tracking->depth_class->width = tracking->depth_cloud->width / seg_res;
        tracking->depth_class->height = tracking->depth_cloud->height / seg_res;
        tracking->depth_class->points.resize(tracking->depth_class->width *
                                             tracking->depth_class->height);
        tracking->depth_class->is_dense = false;

        int n_lores_points = 0;
        foreach_xy_off(tracking->depth_class->width,
                       tracking->depth_class->height) {
            int hoff = (y * seg_res) * tracking->depth_cloud->width +
                (x * seg_res);
            tracking->depth_class->points[off].x =
                tracking->depth_cloud->points[hoff].x;
            tracking->depth_class->points[off].y =
                tracking->depth_cloud->points[hoff].y;
            tracking->depth_class->points[off].z =
                tracking->depth_cloud->points[hoff].z;
            tracking->depth_class->points[off].label = -1;
            if (!std::isnan(tracking->depth_class->points[off].z)) {
                ++n_lores_points;
            }
        }
    }
}

static void
stage_downsample_debug_cb(struct gm_tracking_impl *tracking,
                          struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_class);
    colour_debug_cloud(ctx, tracking, tracking->depth_class, false);
}

static void
stage_ground_project_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    glm::mat4 to_ground = state->to_ground;

    // Transform the cloud into ground-aligned space if we have a valid pose
    if (!tracking->ground_cloud) {
        tracking->ground_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(
            new pcl::PointCloud<pcl::PointXYZL>);
    }
    if (ctx->codebook_pose.type != GM_POSE_INVALID) {
        unsigned depth_class_size = tracking->depth_class->points.size();

        tracking->ground_cloud->width = tracking->depth_class->width;
        tracking->ground_cloud->height = tracking->depth_class->height;
        tracking->ground_cloud->points.resize(depth_class_size);
        tracking->ground_cloud->is_dense = false;

        float nan = std::numeric_limits<float>::quiet_NaN();
        pcl::PointXYZL invalid_pt;
        invalid_pt.x = invalid_pt.y = invalid_pt.z = nan;
        invalid_pt.label = -1;

        foreach_xy_off(tracking->depth_class->width,
                       tracking->depth_class->height)
        {
            pcl::PointXYZL &point =
                tracking->depth_class->points[off];
            if (std::isnan(point.z)) {
                tracking->ground_cloud->points[off] = invalid_pt;
                continue;
            }

            glm::vec4 pt(point.x, point.y, point.z, 1.f);
            pt = (to_ground * pt);

            tracking->ground_cloud->points[off].x = pt.x;
            tracking->ground_cloud->points[off].y = pt.y;
            tracking->ground_cloud->points[off].z = pt.z;
        }
    } else {
        tracking->ground_cloud->resize(0);
    }
}

static void
stage_ground_project_debug_cb(struct gm_tracking_impl *tracking,
                              struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    if (ctx->codebook_pose.type != GM_POSE_INVALID) {
        add_debug_cloud_xyz_from_pcl_xyzl_transformed(ctx, tracking,
                                                      tracking->depth_class,
                                                      state->to_ground);
        colour_debug_cloud(ctx, tracking, tracking->depth_class, false);
    } else {
        add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_class);
        colour_debug_cloud(ctx, tracking, tracking->depth_class, false);
    }
}

static void
stage_codebook_project_debug_cb(struct gm_tracking_impl *tracking,
                                struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    glm::mat4 to_start = state->to_start;
    glm::mat4 start_to_codebook = ctx->start_to_codebook;
    int seg_res = ctx->seg_res;

    add_debug_cloud_xyz_of_codebook_space(
        ctx, tracking, tracking->depth_class, to_start,
        start_to_codebook, &tracking->depth_camera_intrinsics, seg_res);
    colour_debug_cloud(ctx, tracking, tracking->depth_class, false);
}

static void
stage_codebook_classify_cb(struct gm_tracking_impl *tracking,
                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    glm::mat4 to_start = state->to_start;
    glm::mat4 start_to_codebook = ctx->start_to_codebook;
    unsigned depth_class_size = tracking->depth_class->points.size();
    int seg_res = ctx->seg_res;

    // Remove depth classification old codewords
    for (unsigned off = 0; off < ctx->depth_seg.size(); ++off) {
        std::vector<struct seg_codeword> &codewords = ctx->depth_seg[off];
        ctx->depth_seg_bg[off] = NULL;
        for (unsigned i = 0; i < codewords.size();) {
            struct seg_codeword &codeword = codewords[i];
            if ((tracking->frame->timestamp - codeword.tl) / 1000000000.0 >=
                ctx->seg_timeout)
            {
                std::swap(codeword, codewords.back());
                codewords.pop_back();
            } else {
                if (!ctx->depth_seg_bg[off] ||
                    codeword.n > ctx->depth_seg_bg[off]->n) {
                    ctx->depth_seg_bg[off] = &codeword;
                }
                ++i;
            }
        }
    }

    // Do classification of depth buffer
    for (unsigned depth_off = 0; depth_off < depth_class_size; ++depth_off)
    {
        pcl::PointXYZL point =
            tracking->depth_class->points[depth_off];

        if (std::isnan(point.z)) {
            // We'll never cluster a nan value, so we can immediately
            // classify it as background.
            tracking->depth_class->points[depth_off].label = BG;
            continue;
        } else {
            int off = project_point_into_codebook(
                &point, to_start, start_to_codebook,
                &tracking->depth_camera_intrinsics, seg_res);

            // Falls outside of codebook so we can't classify...
            if (off < 0)
                continue;

            // At this point z has been projected into the coordinate space
            // of the codebook
            float depth = point.z;

            const uint64_t t = tracking->frame->timestamp;
            const float tb = ctx->seg_tb;
            const float tf = ctx->seg_tf;
            const int b = ctx->seg_b;
            const int gamma = (float)ctx->seg_gamma;
            const int alpha = ctx->seg_alpha;
            const float psi = ctx->seg_psi;

            // Look to see if this pixel falls into an existing codeword
            struct seg_codeword *codeword = NULL;
            struct seg_codeword *bg_codeword = ctx->depth_seg_bg[off];

            std::vector<struct seg_codeword> &codewords =
                ctx->depth_seg[off];
            for (std::vector<struct seg_codeword>::iterator it =
                 codewords.begin(); it != codewords.end(); ++it) {
                struct seg_codeword &candidate = *it;

                float dist = fabsf(depth - candidate.m);
                if (dist < tb) {
                    codeword = &candidate;
                    break;
                }
            }

            assert(bg_codeword || (!bg_codeword && !codeword));

            // Classify this depth value
            const float frame_time = ctx->n_tracking ?
                (float)(t - ctx->tracking_history[0]->frame->timestamp) :
                100000000.f;

            if (!codeword) {
                tracking->depth_class->points[depth_off].label = FG;
            } else if (codeword->n == bg_codeword->n) {
                tracking->depth_class->points[depth_off].label = BG;
            } else {
                bool flat = false, flickering = false;
                float mean_diff = fabsf(codeword->m - bg_codeword->m);
                if ((tb < mean_diff) && (mean_diff <= tf)) {
                    flat = true;
                }
                if ((b * codeword->nc) > codeword->n &&
                    (int)(((t - codeword->ts) / frame_time) / gamma) <=
                    codeword->nc) {
                    flickering = true;
                }
                if (flat || flickering) {
                    tracking->depth_class->points[depth_off].label =
                        (flat && flickering) ?
                            FL_FLK : (flat ?  FL : FLK);
                } else {
                    if (codeword->n > alpha &&
                        ((codeword->tl - codeword->ts) / frame_time) /
                        (float)codeword->n >= psi) {
                        tracking->depth_class->points[depth_off].label = TB;
                    } else {
                        tracking->depth_class->points[depth_off].label = FG;
                    }
                }
            }
        }
    }
}

static void
stage_codebook_classify_debug_cb(struct gm_tracking_impl *tracking,
                                 struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_class);
    colour_debug_cloud(ctx, tracking, tracking->depth_class,
                       true); // classification done
}

static void
stage_naive_detect_floor_cb(struct gm_tracking_impl *tracking,
                            struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    unsigned depth_class_size = tracking->depth_class->points.size();
    enum tracking_stage debug_stage_id = (enum tracking_stage)ctx->debug_pipeline_stage;

    // If we've not tracked a human yet, the depth classification may not
    // be reliable - just use a simple clustering technique to find a
    // human and separate them from the floor, then rely on motion detection
    // for subsequent frames.
    int width = (int)tracking->depth_class->width;
    int height = (int)tracking->depth_class->height;
    int fx = width / 2;
    int fy = height / 2;
    int fidx = fy * width + fx;

    // First search a small box in the center of the image and pick the
    // nearest point to start our flood-fill from.
    int fw = width / 8; // TODO: make box size configurable
    int fh = height / 8;

    float fz = FLT_MAX;
    int fr_i = 0;
    struct focal_point {
        float fz;
        int idx;
    } focal_region[fw * fh];

    int x0 = fx - fw / 2;
    int y0 = fy - fh / 2;
    for (int y = y0; y < (y0 + fh); y++) {
        for (int x = x0; x < (x0 + fw); x++) {
            int idx = y * width + x;
            pcl::PointXYZL &point =
                tracking->depth_class->points[idx];
            if (!std::isnan(point.z))
                focal_region[fr_i++] = { point.z, idx };
        }
    }

    //gm_assert(ctx->log, fr_i == fw * fh, "Flibble");
    // Use the median point as our focal point
    //
    // XXX: We tried a simpler approach of selecting the nearest point
    // previously but with noisy data we would somtimes select a
    // disconnected point that then wouldn't cluster with anything
    //
    // XXX: we could calculate the median more optimally if this
    // shows up in any profiling...
    std::sort(focal_region, focal_region + fr_i,
              [](focal_point a, focal_point b) { return a.fz < b.fz; });
    fr_i /= 2;

    int idx = focal_region[fr_i].idx;
    fx = idx % width;
    fy = idx / width;
    fz = focal_region[fr_i].fz;

    if (ctx->debug_cloud_mode) {
        // Draw the lines of focus...
        if (fz != FLT_MAX) {
            float line_x = tracking->depth_class->points[focal_region[fr_i].idx].x;
            float line_y = tracking->depth_class->points[focal_region[fr_i].idx].y;
            tracking_draw_line(tracking,
                               0, 0, 0,
                               0, 0, 4,
                               0x808080ff);
            tracking_draw_line(tracking,
                               0, 0, 0,
                               line_x, line_y, fz,
                               0x00ff00ff);
        } else {
            tracking_draw_line(tracking,
                               0, 0, 0,
                               0, 0, 4,
                               0xff0000ff);
        }
    }

    state->naive_fx = fx;
    state->naive_fy = fy;

    // Flood-fill downwards from the focal point, with a limit on the x and
    // z axes for how far a point can be from the focus. This will allow
    // us to hopefully find the floor level and establish a y limit before
    // then flood-filling again without the x and z limits.

    std::queue<struct PointCmp> &flood_fill = state->flood_fill;
    flood_fill.push({ fx, fy, fx, fy });

    std::vector<bool> &done_mask = state->done_mask;
    done_mask.resize(depth_class_size, false);

    pcl::PointXYZL &focus_pt =
        tracking->depth_class->points[fidx];

    float lowest_point = -FLT_MAX;
    while (!flood_fill.empty()) {
        struct PointCmp point = flood_fill.front();
        flood_fill.pop();

        int idx = point.y * width + point.x;

        if (point.x < 0 || point.y < fy ||
            point.x >= width || point.y >= height ||
            done_mask[idx]) {
            continue;
        }

        pcl::PointXYZL &pcl_pt =
            tracking->depth_class->points[idx];

        if (fabsf(focus_pt.x - pcl_pt.x) > ctx->cluster_max_width ||
            fabsf(focus_pt.z - pcl_pt.z) > ctx->cluster_max_depth) {
            continue;
        }

        float aligned_y = (ctx->codebook_pose.type != GM_POSE_INVALID) ?
            tracking->ground_cloud->points[idx].y :
            tracking->depth_class->points[idx].y;
        if (aligned_y > lowest_point) {
            lowest_point = aligned_y;
        }

        if (gm_compare_depth(tracking->depth_class,
                             point.x, point.y, point.lx, point.ly,
                             ctx->cluster_tolerance))
        {
            done_mask[idx] = true;
            flood_fill.push({ point.x - 1, point.y, point.x, point.y });
            flood_fill.push({ point.x + 1, point.y, point.x, point.y });
            flood_fill.push({ point.x, point.y - 1, point.x, point.y });
            flood_fill.push({ point.x, point.y + 1, point.x, point.y });

            // TODO: move outside loop, and instead iterate flood_fill
            // queue when done
            if (debug_stage_id == TRACKING_STAGE_NAIVE_FLOOR &&
                ctx->debug_cloud_mode)
            {
                struct gm_point_rgba debug_point;

                debug_point.x = tracking->depth_class->points[idx].x;
                debug_point.y = -tracking->depth_class->points[idx].y; // FIXME
                debug_point.z = tracking->depth_class->points[idx].z;
                debug_point.rgba = 0xffffffff;

                tracking->debug_cloud.push_back(debug_point);
                tracking->debug_cloud_indices.push_back(idx);
            }
        }
    }

    state->naive_floor_y = lowest_point;
}

static void
stage_naive_detect_floor_debug_cb(struct gm_tracking_impl *tracking,
                                  struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    // Note: the actual debug cloud is updated as part of
    // stage_naive_cluster_cb above, so we just need the color..
    colour_debug_cloud(ctx, tracking, tracking->depth_class, true);
}

static void
stage_naive_cluster_cb(struct gm_tracking_impl *tracking,
                       struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    enum tracking_stage debug_stage_id = (enum tracking_stage)ctx->debug_pipeline_stage;

    int width = (int)tracking->depth_class->width;
    int height = (int)tracking->depth_class->height;

    int fx = state->naive_fx;
    int fy = state->naive_fy;

    int fidx = fy * width + fx;
    pcl::PointXYZL &focus_pt = tracking->depth_class->points[fidx];

    float lowest_point = state->naive_floor_y;

    std::queue<struct PointCmp> &flood_fill = state->flood_fill;
    std::vector<bool> &done_mask = state->done_mask;

    flood_fill.push({ fx, fy, fx, fy });
    std::fill(done_mask.begin(), done_mask.end(), false);

    pcl::PointIndices person_indices;

    while (!flood_fill.empty()) {
        struct PointCmp point = flood_fill.front();
        flood_fill.pop();

        int idx = point.y * width + point.x;

        if (point.x < 0 || point.y < 0 ||
            point.x >= width || point.y >= height ||
            done_mask[idx]) {
            continue;
        }

        pcl::PointXYZL &pcl_pt =
            tracking->depth_class->points[idx];

        // Avoid building a cloud that would be considered invalid. We
        // assume the focus point is somewhere near the center of the body,
        // but not the exact center (so we divide by 1.75 and not 2).
        if (fabsf(focus_pt.x - pcl_pt.x) > ctx->cluster_max_width / 1.75f ||
            fabsf(focus_pt.y - pcl_pt.y) > ctx->cluster_max_height / 1.75f ||
            fabsf(focus_pt.z - pcl_pt.z) > ctx->cluster_max_depth / 1.75f) {
            continue;
        }

        float aligned_y = (ctx->codebook_pose.type != GM_POSE_INVALID) ?
            tracking->ground_cloud->points[idx].y :
            tracking->depth_class->points[idx].y;
        if (aligned_y > lowest_point - ctx->floor_threshold) {
            continue;
        }

        if (gm_compare_depth(tracking->depth_class,
                             point.x, point.y, point.lx, point.ly,
                             ctx->cluster_tolerance)) {
            done_mask[idx] = true;
            person_indices.indices.push_back(idx);
            flood_fill.push({ point.x - 1, point.y, point.x, point.y });
            flood_fill.push({ point.x + 1, point.y, point.x, point.y });
            flood_fill.push({ point.x, point.y - 1, point.x, point.y });
            flood_fill.push({ point.x, point.y + 1, point.x, point.y });

            // TODO: move outside loop, and instead iterate flood_fill
            // queue when done
            if (debug_stage_id == TRACKING_STAGE_NAIVE_CLUSTER &&
                ctx->debug_cloud_mode)
            {
                struct gm_point_rgba debug_point;

                debug_point.x = tracking->depth_class->points[idx].x;
                debug_point.y = -tracking->depth_class->points[idx].y; // FIXME
                debug_point.z = tracking->depth_class->points[idx].z;
                debug_point.rgba = 0xffffffff;

                tracking->debug_cloud.push_back(debug_point);
                tracking->debug_cloud_indices.push_back(idx);
            }
        }
    }

    if (!person_indices.indices.empty()) {
        std::vector<pcl::PointIndices> &cluster_indices = state->cluster_indices;
        cluster_indices.push_back(person_indices);
    }
}

static void
stage_naive_cluster_debug_cb(struct gm_tracking_impl *tracking,
                             struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    // Note: the actual debug cloud is updated as part of
    // stage_naive_cluster_cb above, so we just need the color..
    colour_debug_cloud(ctx, tracking, tracking->depth_class, true);
}

static void
stage_codebook_cluster_cb(struct gm_tracking_impl *tracking,
                          struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = state->cluster_indices;

    LabelComparator<pcl::PointXYZL>::Ptr label_cluster(
        new LabelComparator<pcl::PointXYZL>);
    label_cluster->setInputCloud(tracking->depth_class);
    label_cluster->setDepthThreshold(ctx->cluster_tolerance);

    tracking->cluster_labels =
        pcl::PointCloud<pcl::Label>::Ptr(new pcl::PointCloud<pcl::Label>);
    pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZL, pcl::Label>
        depth_connector(label_cluster);
    depth_connector.setInputCloud(tracking->depth_class);
    depth_connector.segment(*tracking->cluster_labels, cluster_indices);
}

static void
stage_codebook_cluster_debug_cb(struct gm_tracking_impl *tracking,
                                struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;

}

static void
stage_filter_clusters_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = state->cluster_indices;

    // Assume the any cluster that has roughly human dimensions and
    // contains its centroid may be a person.

    //const float centroid_tolerance = 0.1f;
    std::vector<pcl::PointIndices> &persons = state->persons;
    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        pcl::PointIndices &points = cluster_indices[i];

        // Check if the cluster has human-ish dimensions
        Eigen::Vector4f min, max;
        pcl::getMinMax3D(*tracking->depth_class, points, min, max);
        Eigen::Vector4f diff = max - min;
        if (diff[0] < ctx->cluster_min_width ||
            diff[0] > ctx->cluster_max_width ||
            diff[1] < ctx->cluster_min_height ||
            diff[1] > ctx->cluster_max_height ||
            diff[2] < ctx->cluster_min_depth ||
            diff[2] > ctx->cluster_max_depth) {
            continue;
        }
        gm_info(ctx->log,
                "Cluster with %d points, (%.2fx%.2fx%.2f)\n",
                (int)(points).indices.size(), diff[0], diff[1], diff[2]);

#if 0
        // Work out the centroid of the cloud and see if there's a point
        // near there. A human, unless they're falling, ought to contain
        // their center of gravity. If they're jumping or falling, we can
        // probably interpolate joint positions.
        // Note that I guess humans are actually quite frequently in a state
        // of semi-falling, so we have a pretty generous tolerance.
        Eigen::VectorXf centroid;
        pcl::computeNDCentroid(*tracking->depth_class, points,
                               centroid);

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
        if (std::isnan(tracking->depth_cloud->points[off].z) ||
            fabsf(centroid[2] - tracking->depth_cloud->points[off].z) >
            centroid_tolerance) {
            continue;
        }
#endif

        persons.push_back(points);
    }
}

static void
stage_filter_clusters_debug_cb(struct gm_tracking_impl *tracking,
                              struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_class);
    colour_debug_cloud(ctx, tracking, tracking->depth_class,
                       true); // classification done
}

static void
stage_project_clusters_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    int seg_res = ctx->seg_res;

    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;

    std::vector<float*> &depth_images = state->depth_images;

    std::vector<pcl::PointIndices> &persons = state->persons;

    for (auto &person : persons) {
        float *depth_img = (float *)xmalloc(width * height * sizeof(float));
        for (int i = 0; i < width * height; ++i) {
            depth_img[i] = HUGE_DEPTH;
        }

        for (int idx : person.indices) {
            int lx = idx % tracking->depth_class->width;
            int ly = idx / tracking->depth_class->width;
            for (int hy = (int)(ly * seg_res), ey = 0;
                 hy < (int)tracking->depth_cloud->height && ey < seg_res;
                 ++hy, ++ey) {
                for (int hx = (int)(lx * seg_res), ex = 0;
                     hx < (int)tracking->depth_cloud->width &&
                     ex < seg_res;
                     ++hx, ++ex) {
                    int off = hy * tracking->depth_cloud->width + hx;

                    // Reproject this point into training camera space
                    glm::vec3 point_t(tracking->depth_cloud->points[off].x,
                                      tracking->depth_cloud->points[off].y,
                                      tracking->depth_cloud->points[off].z);

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
}

static void
stage_project_clusters_debug_cb(struct gm_tracking_impl *tracking,
                               struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_class);
    colour_debug_cloud(ctx, tracking, tracking->depth_class,
                       true); // classification done
}

static void
stage_label_inference_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;

    infer_labels(ctx->log,
                 ctx->decision_trees,
                 ctx->n_decision_trees,
                 state->depth_image,
                 width, height,
                 state->label_probs,
                 true, // multi-threaded
                 ctx->flip_labels);
}

static void
stage_joint_weights_cb(struct gm_tracking_impl *tracking,
                       struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;

    calc_pixel_weights(state->depth_image,
                       state->label_probs,
                       width, height,
                       ctx->n_labels,
                       ctx->joint_map,
                       state->weights);
}

static void
stage_joint_inference_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;

    float vfov =  pcl::rad2deg(2.0f * atanf(0.5 * height /
                               tracking->training_camera_intrinsics.fy));

    state->joints_candidate =
            infer_joints_fast(state->depth_image,
                              state->label_probs,
                              state->weights,
                              width, height,
                              ctx->n_labels,
                              ctx->joint_map,
                              vfov,
                              ctx->joint_params->joint_params);
}

static void
stage_refine_skeleton_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    build_skeleton(ctx, state->joints_candidate, state->candidate_skeleton);
    state->candidate_skeleton.timestamp = tracking->frame->timestamp;

    build_bones(ctx, state->candidate_skeleton);
    refine_skeleton(ctx, state->joints_candidate, state->candidate_skeleton);
}

static void
stage_sanitize_skeleton_cb(struct gm_tracking_impl *tracking,
                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    sanitise_skeleton(ctx, tracking->skeleton,
                      tracking->frame->timestamp);
}

static void
add_debug_cloud_person_masks_except(struct gm_tracking_impl *tracking,
                                    struct pipeline_scratch_state *state,
                                    int except_person)

{
    struct gm_context *ctx = tracking->ctx;

    /* Also show other failed candidates... */
    for (int i = 0; i < (int)state->persons.size(); i++) {
        if (i == except_person)
            continue;
        add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                      tracking->depth_class,
                                                      state->persons[i].indices);
    }
}

static void
add_best_person_depth_image_debug_cb(struct gm_tracking_impl *tracking,
                                      struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    float *depth_img = state->depth_images[state->best_person];

    add_debug_cloud_xyz_from_dense_depth_buf(ctx, tracking,
                                             depth_img,
                                             &tracking->training_camera_intrinsics);
    colour_debug_cloud(ctx, tracking, NULL, false);

    add_debug_cloud_person_masks_except(tracking, state, state->best_person);
}

static void
stage_select_best_person_cloud_debug_cb(struct gm_tracking_impl *tracking,
                                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->depth_class,
                                                  state->persons[state->best_person].indices);
    colour_debug_cloud(ctx, tracking, tracking->depth_class, true);

    add_debug_cloud_person_masks_except(tracking, state, state->best_person);
}

static void
stage_update_codebook_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    update_depth_codebook(ctx, tracking,
                          state->to_start, ctx->start_to_codebook,
                          ctx->seg_res);
}

#if 0
static void
stage__cb(struct gm_tracking_impl *tracking,
                struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;

}

static void
stage__debug_cb(struct gm_tracking_impl *tracking,
                      struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;

}
#endif

static void
run_stage_debug(struct gm_tracking_impl *tracking,
                enum tracking_stage stage_id,
                void (*stage_debug_callback)(struct gm_tracking_impl *tracking,
                                             struct pipeline_scratch_state *state),
                struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    if (ctx->debug_cloud_mode &&
        ctx->debug_pipeline_stage == stage_id &&
        stage_debug_callback)
    {
        stage_debug_callback(tracking, state);
    }
}

static void
run_stage(struct gm_tracking_impl *tracking,
          enum tracking_stage stage_id,
          void (*stage_callback)(struct gm_tracking_impl *tracking,
                                 struct pipeline_scratch_state *state),
          void (*stage_debug_callback)(struct gm_tracking_impl *tracking,
                                       struct pipeline_scratch_state *state),
          struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    struct gm_pipeline_stage &stage = ctx->stages[stage_id];
    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_id];

    uint64_t duration;

    if (stage_callback) {
        uint64_t start = get_time();

        stage_callback(tracking, state);

        uint64_t end = get_time();
        duration = end - start;
    } else
        duration = 0;

    // Note: we use += here because a stage may be used multiple times
    // during tracking and we reset the per-stage durations when we
    // start gm_context_track_skeleton()
    stage_data.duration_ns += duration;

    stage.total_time_ns += duration;
    stage.n_invocations++;

    gm_info(ctx->log,
            "Stage: %s took %.3f%s",
            stage.name,
            get_duration_ns_print_scale(duration),
            get_duration_ns_print_scale_suffix(duration));

    run_stage_debug(tracking, stage_id, stage_debug_callback, state);
}

static bool
gm_context_track_skeleton(struct gm_context *ctx,
                          struct gm_tracking_impl *tracking)
{
    uint64_t start, end, duration;
    enum tracking_stage debug_stage_id = (enum tracking_stage)ctx->debug_pipeline_stage;
    struct pipeline_scratch_state state(ctx->n_joints);

    for (int i = 0; i < tracking->stage_data.size(); i++) {
        tracking->stage_data[i].duration_ns = 0;
    }

    if (ctx->debug_cloud_mode) {
        tracking->debug_cloud.resize(0);
        tracking->debug_cloud_indices.resize(0);
        tracking->debug_lines.resize(0);
    }

    float nan = std::numeric_limits<float>::quiet_NaN();
    pcl::PointXYZL invalid_pt;
    invalid_pt.x = invalid_pt.y = invalid_pt.z = nan;
    invalid_pt.label = -1;

    // X increases to the right
    // Y increases downwards
    // Z increases outwards

    if (debug_stage_id == TRACKING_STAGE_START &&
        ctx->debug_cloud_mode)
    {
        add_debug_cloud_xyz_from_dense_depth_buf(ctx, tracking,
                                                 tracking->depth,
                                                 &tracking->depth_camera_intrinsics);
        colour_debug_cloud(ctx, tracking, NULL, false);
    }

    run_stage(tracking,
              TRACKING_STAGE_GAP_FILLED,
              stage_gap_fill_cb,
              stage_gap_fill_debug_cb,
              &state);

    run_stage(tracking,
              TRACKING_STAGE_DOWNSAMPLED,
              stage_downsample_cb,
              stage_downsample_debug_cb,
              &state);

    unsigned depth_class_size = tracking->depth_class->points.size();
    bool motion_detection = ctx->motion_detection;


    ctx->reset_codebook = false;

    if (ctx->depth_seg.size() != depth_class_size ||
        (ctx->codebook_pose.type != tracking->frame->pose.type))
    {
        gm_debug(ctx->log, "XXX: Resetting pose");
        ctx->reset_codebook = true;
    } else if (tracking->frame->pose.type == GM_POSE_TO_START) {
        // Check if the angle or distance between the current frame and the
        // reference frame exceeds a certain threshold, and in that case,
        // reset motion tracking.
        float angle = glm::degrees(glm::angle(
            glm::normalize(glm::quat(ctx->codebook_pose.orientation[3],
                                     ctx->codebook_pose.orientation[0],
                                     ctx->codebook_pose.orientation[1],
                                     ctx->codebook_pose.orientation[2])) *
            glm::inverse(glm::normalize(glm::quat(
                tracking->frame->pose.orientation[3],
                tracking->frame->pose.orientation[0],
                tracking->frame->pose.orientation[1],
                tracking->frame->pose.orientation[2])))));
        while (angle > 180.f) angle -= 360.f;

        float distance = glm::distance(
            glm::vec3(ctx->codebook_pose.translation[0],
                      ctx->codebook_pose.translation[1],
                      ctx->codebook_pose.translation[2]),
            glm::vec3(tracking->frame->pose.translation[0],
                      tracking->frame->pose.translation[1],
                      tracking->frame->pose.translation[2]));

        gm_debug(ctx->log, "XXX: Angle: %.2f, "
                 "Distance: %.2f (%.2f, %.2f, %.2f)", angle, distance,
                 tracking->frame->pose.translation[0] -
                 ctx->codebook_pose.translation[0],
                 tracking->frame->pose.translation[1] -
                 ctx->codebook_pose.translation[1],
                 tracking->frame->pose.translation[2] -
                 ctx->codebook_pose.translation[2]);
        if (angle > 10.f || distance > 0.3f) {
            // We've strayed too far from the initial pose, reset
            // segmentation and use this as the home pose.
            gm_debug(ctx->log, "XXX: Resetting pose (moved too much)");
            ctx->reset_codebook = true;
        }
    }

    state.to_start = glm::mat4(1.0);
    state.to_ground = glm::mat4(1.0);
    switch (tracking->frame->pose.type) {
    case GM_POSE_INVALID:
        break;
    case GM_POSE_TO_START:
        state.to_start = pose_to_matrix(tracking->frame->pose);
        state.to_ground = state.to_start;
        break;
    case GM_POSE_TO_GROUND:
        state.to_ground = pose_to_matrix(tracking->frame->pose);
        break;
    }

    if (ctx->reset_codebook) {
        ctx->depth_seg.clear();
        ctx->depth_seg.resize(depth_class_size);
        ctx->depth_seg_bg.resize(depth_class_size);
        ctx->codebook_pose = tracking->frame->pose;
        ctx->start_to_codebook = glm::inverse(state.to_start);

        if (tracking->frame->pose.type != GM_POSE_TO_START)
            gm_debug(ctx->log, "XXX: No tracking pose");
    }

    run_stage(tracking,
              TRACKING_STAGE_GROUND_SPACE,
              stage_ground_project_cb,
              stage_ground_project_debug_cb,
              &state);

    if (motion_detection) {
        // This is only a logical debug stage, since the projection
        // is actually combined with the classification
        run_stage(tracking,
                  TRACKING_STAGE_CODEBOOK_SPACE,
                  NULL, // no real work to do
                  stage_codebook_project_debug_cb,
                  &state);

        run_stage(tracking,
                  TRACKING_STAGE_CODEBOOK_CLASSIFY,
                  stage_codebook_classify_cb,
                  stage_codebook_classify_debug_cb,
                  &state);
    }


    if (!motion_detection ||
        !ctx->latest_tracking ||
        !ctx->latest_tracking->success ||
        ctx->reset_codebook)
    {
        run_stage(tracking,
                  TRACKING_STAGE_NAIVE_FLOOR,
                  stage_naive_detect_floor_cb,
                  stage_naive_detect_floor_debug_cb,
                  &state);

        run_stage(tracking,
                  TRACKING_STAGE_NAIVE_CLUSTER,
                  stage_naive_cluster_cb,
                  stage_naive_cluster_debug_cb,
                  &state);
    } else {
        run_stage(tracking,
                  TRACKING_STAGE_CODEBOOK_CLUSTER,
                  stage_codebook_cluster_cb,
                  stage_codebook_cluster_debug_cb,
                  &state);
    }

    run_stage(tracking,
              TRACKING_STAGE_FILTER_CLUSTERS,
              stage_filter_clusters_cb,
              stage_filter_clusters_debug_cb,
              &state);

    if (state.persons.size() == 0) {
        if (ctx->motion_detection) {
            update_depth_codebook(ctx, tracking,
                                  state.to_start,
                                  ctx->start_to_codebook,
                                  ctx->seg_res);
        }
        gm_info(ctx->log, "Skipping detection: Could not find a person cluster");
        return false;
    }

    run_stage(tracking,
              TRACKING_STAGE_PROJECT_CLUSTERS,
              stage_project_clusters_cb,
              stage_project_clusters_debug_cb,
              &state);


    int width = tracking->training_camera_intrinsics.width;
    int height = tracking->training_camera_intrinsics.height;
    std::vector<float*> &depth_images = state.depth_images;
    state.weights = (float*)
        xmalloc(width * height * ctx->n_joints * sizeof(float));
    state.label_probs = (float*)xmalloc(width * height * ctx->n_labels *
                                         sizeof(float));
    state.best_person = 0;
    for (int i = 0; i < (int)depth_images.size(); i++) {
        state.depth_image = depth_images[i];

        run_stage(tracking,
                  TRACKING_STAGE_LABEL_INFERENCE,
                  stage_label_inference_cb,
                  NULL,
                  &state);

        run_stage(tracking,
                  TRACKING_STAGE_JOINT_WEIGHTS,
                  stage_joint_weights_cb,
                  NULL,
                  &state);

        run_stage(tracking,
                  TRACKING_STAGE_JOINT_INFERENCE,
                  stage_joint_inference_cb,
                  NULL,
                  &state);
        assert(state.joints_candidate->n_joints == ctx->n_joints);

        run_stage(tracking,
                  TRACKING_STAGE_REFINE_SKELETON,
                  stage_refine_skeleton_cb,
                  NULL,
                  &state);

        // If this skeleton has higher confidence than the last, keep it
        if (i == 0 ||
            compare_skeletons(state.candidate_skeleton, tracking->skeleton))
        {
            std::swap(tracking->skeleton, state.candidate_skeleton);
            std::swap(tracking->label_probs, state.label_probs);
            state.best_person = i;
        }

        free_joints(state.joints_candidate);
        state.joints_candidate = NULL;
    }
    xfree(state.label_probs);
    state.label_probs = NULL;
    xfree(state.weights);
    state.weights = NULL;

    // Only a logical stage since selection is part of the
    // iteration above but we want to see the point cloud
    // view of the best_person candidate before they were
    // projected into a depth image used for label
    // inference...
    run_stage_debug(tracking,
                    TRACKING_STAGE_SELECT_CANDIDATE_CLUSTER,
                    stage_select_best_person_cloud_debug_cb,
                    &state);

    // We want the cloud view for these stages to represent the
    // depth_image of the best_person candidate, so we wait until
    // we've processed all candiates before updating the debug
    // state...
    //
    // TODO: each stage should have a custom callback to e.g.
    // handle colouring the points according to inferred labels
    run_stage_debug(tracking,
                    TRACKING_STAGE_LABEL_INFERENCE,
                    add_best_person_depth_image_debug_cb,
                    &state);
    run_stage_debug(tracking,
                    TRACKING_STAGE_JOINT_WEIGHTS,
                    add_best_person_depth_image_debug_cb,
                    &state);
    run_stage_debug(tracking,
                    TRACKING_STAGE_JOINT_INFERENCE,
                    add_best_person_depth_image_debug_cb,
                    &state);
    run_stage_debug(tracking,
                    TRACKING_STAGE_REFINE_SKELETON,
                    add_best_person_depth_image_debug_cb,
                    &state);

    for (unsigned i = 0; i < depth_images.size(); ++i) {
        xfree(depth_images[i]);
        depth_images[i] = NULL;
    }
    depth_images.clear();

    bool tracked = !ctx->skeleton_validation ||
        (tracking->skeleton.confidence >= ctx->skeleton_min_confidence &&
         tracking->skeleton.distance <= ctx->skeleton_max_distance);

    // Update the depth classification so it knows which pixels are tracked
    // TODO: We should actually use the label cluster points, which may not
    //       consist of this entire cloud.
    std::vector<pcl::PointIndices> &persons = state.persons;
    pcl::PointIndices &person = persons[state.best_person];
    int tracked_label = tracked ? TRK : CAN;
    for (std::vector<int>::const_iterator it = person.indices.begin();
         it != person.indices.end(); ++it) {
        tracking->depth_class->points[*it].label = tracked_label;
    }


    if (tracked) {
        start = get_time();

        // The codebook may have been polluted before this point by a human
        // failing to track. To counteract this, as well as removing and of
        // the codewords that apply to the tracked figure, we also remove all
        // but the furthest-away codewords. This is in the hope that if there
        // was an untracked human in the codebook that at some point we saw
        // background behind them.
        if (motion_detection &&
            (!ctx->latest_tracking || !ctx->latest_tracking->success)) {
#if 0
            foreach_xy_off(tracking->depth_class->width,
                           tracking->depth_class->height) {
                std::list<struct seg_codeword> &codewords = ctx->depth_seg[off];
                if (codewords.size() <= 1) {
                    continue;
                }

                std::list<struct seg_codeword>::iterator furthest, it;
                for (furthest = it = codewords.begin();
                     it != codewords.end(); ++it) {
                    if ((*it).m > (*furthest).m) {
                        furthest = it;
                    }
                }

                it = codewords.begin();
                while (it != codewords.end()) {
                    if (it == furthest) {
                        ++it;
                    } else {
                        it = codewords.erase(it);
                    }
                }
            }
#else
            ctx->depth_seg.clear();
            ctx->depth_seg.resize(depth_class_size);
            ctx->depth_seg_bg.resize(depth_class_size);
#endif
        }

        // TODO: We just take the most confident skeleton above, but we should
        //       probably establish some thresholds and spit out multiple
        //       skeletons.
        if (ctx->bone_sanitisation) {
            run_stage(tracking,
                      TRACKING_STAGE_SANITIZE_SKELETON,
                      stage_sanitize_skeleton_cb,
                      NULL,
                      &state);
        }
        for (int j = 0; j < ctx->n_joints; j++) {
            int idx = j * 3;
            tracking->joints_processed[idx] = tracking->skeleton.joints[j].x;
            tracking->joints_processed[idx+1] = tracking->skeleton.joints[j].y;
            tracking->joints_processed[idx+2] = tracking->skeleton.joints[j].z;
        }

        end = get_time();
        duration = end - start;
        LOGI("Joint processing took %.3f%s",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));

        if (motion_detection) {
            run_stage(tracking,
                      TRACKING_STAGE_UPDATE_CODEBOOK,
                      stage_update_codebook_cb,
                      NULL,
                      &state);
        }
    }

    return tracked;
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
    case GM_FORMAT_BGR_U8:
        foreach_xy_off(width, height) {
            tracking->face_detect_buf[off] = RGB2Y(video[off * 3 + 2],
                                                   video[off * 3 + 1],
                                                   video[off * 3]);
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
    case GM_FORMAT_BGRX_U8:
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 4 + 2];
            uint8_t g = video[off * 4 + 1];
            uint8_t b = video[off * 4];
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
    case GM_FORMAT_BGRA_U8:
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 4 + 2];
            uint8_t g = video[off * 4 + 1];
            uint8_t b = video[off * 4];
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
                             struct gm_intrinsics *frame_intrinsics,
                             enum gm_format format,
                             struct gm_buffer *buffer)
{
    int width = frame_intrinsics->width;
    int height = frame_intrinsics->height;
    int rot_width = tracking->depth_camera_intrinsics.width;
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

        float fx = frame_intrinsics->fx;
        float fy = frame_intrinsics->fy;
        float cx = frame_intrinsics->cx;
        float cy = frame_intrinsics->cy;

        float k1, k2, k3;
        k1 = k2 = k3 = 0.f;

        /* XXX: we only support applying the brown's model... */
        bool apply_distortion = ctx->apply_depth_distortion;
        if (apply_distortion) {
            switch (frame_intrinsics->distortion_model) {
            case GM_DISTORTION_NONE:
                apply_distortion = false;
                break;
            case GM_DISTORTION_FOV_MODEL:
                apply_distortion = false;
                break;
            case GM_DISTORTION_BROWN_K1_K2:
                k1 = frame_intrinsics->distortion[0];
                k2 = frame_intrinsics->distortion[1];
                k3 = 0;
                break;
            case GM_DISTORTION_BROWN_K1_K2_K3:
                k1 = frame_intrinsics->distortion[0];
                k2 = frame_intrinsics->distortion[1];
                k3 = frame_intrinsics->distortion[2];
                break;
            case GM_DISTORTION_BROWN_K1_K2_P1_P2_K3:
                k1 = frame_intrinsics->distortion[0];
                k2 = frame_intrinsics->distortion[1];
                k3 = frame_intrinsics->distortion[4];
                /* Ignoring tangential distortion */
                break;
            }
        }

        for (int off = 0; off < num_points; off++) {
            float *xyzc = ((float *)buffer->data) + 4 * off;
            float rd, ru;

            // Reproject this point into training camera space
            glm::vec3 point_t(xyzc[0], xyzc[1], xyzc[2]);

            int x;
            if (apply_distortion) {
                ru = sqrtf((point_t.x*point_t.x + point_t.y*point_t.y) /
                           (point_t.z*point_t.z));

                // Google documented their POLY_3 distortion model should
                // be evaluated as:
                //   rd = ru + k1 * ru^3 + k2 * ru^5 + k3 * ru^7
                // they also refer to the same model as Brown's with only
                // k1, k2 and k3 coefficients, but e.g. referencing Wikipedia
                // and looking for other interpretations of the Brown-Conrady
                // model then it looks like there's some inconsistency with
                // the ru exponents used. Wikipedia uses:
                //   k1 * ru^2 + k2 * ru^4 + k3 * ru^6
#if 0
                float ru2 = ru*ru;
                float ru3 = ru2*ru;
                float ru5 = ru3*ru2;
                float ru7 = ru5*ru2;
                rd = ru + k1 * ru3 + k2 * ru5 + k3 * ru7;
#else
                float ru2 = ru*ru;
                float ru4 = ru2*ru2;
                float ru6 = ru2*ru4;
                rd = ru + k1 * ru2 + k2 * ru4 + k3 * ru6;
#endif

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
                {
                    depth_copy[roff] = point_t.z;
                });
        }
        break;
    }
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_LUMINANCE_U8:
    case GM_FORMAT_RGB_U8:
    case GM_FORMAT_RGBX_U8:
    case GM_FORMAT_RGBA_U8:
    case GM_FORMAT_BGR_U8:
    case GM_FORMAT_BGRX_U8:
    case GM_FORMAT_BGRA_U8:
        gm_assert(ctx->log, 0, "Unexpected format for depth buffer");
        break;
    }
}

static struct gm_tracking_impl *
mem_pool_acquire_tracking(struct gm_mem_pool *pool)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)
        mem_pool_acquire_resource(pool);
    gm_assert(tracking->ctx->log, atomic_load(&tracking->base.ref) == 0,
              "Tracking object in pool with non-zero ref-count");

    atomic_store(&tracking->base.ref, 1);

    tracking->success = false;

    return tracking;
}

static struct gm_prediction_impl *
mem_pool_acquire_prediction(struct gm_mem_pool *pool)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)
        mem_pool_acquire_resource(pool);
    gm_assert(prediction->ctx->log, atomic_load(&prediction->base.ref) == 0,
              "Prediction object in pool with non-zero ref-count");

    atomic_store(&prediction->base.ref, 1);

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

    gm_debug(ctx->log, "Initialising Dlib frontal face detector took %.3f%s",
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

        gm_assert(ctx->log,
                  frame->video_intrinsics.width > 0 &&
                  frame->video_intrinsics.height > 0,
                  "Invalid frame video intrinsics for tracking");
        gm_context_rotate_intrinsics(ctx,
                                     &frame->video_intrinsics,
                                     &tracking->video_camera_intrinsics,
                                     tracking->frame->camera_rotation);

        gm_assert(ctx->log,
                  frame->depth_intrinsics.width > 0 &&
                  frame->depth_intrinsics.height > 0,
                  "Invalid frame depth intrinsics for tracking");
        gm_context_rotate_intrinsics(ctx,
                                     &frame->depth_intrinsics,
                                     &tracking->depth_camera_intrinsics,
                                     tracking->frame->camera_rotation);

        tracking->training_camera_intrinsics = ctx->training_camera_intrinsics;

        copy_and_rotate_depth_buffer(ctx,
                                     tracking,
                                     &frame->depth_intrinsics,
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
        end = get_time();
        duration = end - start;
        gm_debug(ctx->log, "Tracking preparation took %.3f%s",
                 get_duration_ns_print_scale(duration),
                 get_duration_ns_print_scale_suffix(duration));

        start = get_time();
        bool tracked = gm_context_track_skeleton(ctx, tracking);

        end = get_time();
        duration = end - start;
        gm_debug(ctx->log, "Skeletal tracking took %.3f%s",
                 get_duration_ns_print_scale(duration),
                 get_duration_ns_print_scale_suffix(duration));
        tracking->duration_ns = duration;

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
                         atomic_load(&ctx->tracking_history[0]->base.ref));
                gm_tracking_unref(&ctx->tracking_history[0]->base);
            }
            ctx->tracking_history[0] = (struct gm_tracking_impl *)
                gm_tracking_ref(&tracking->base);

            gm_debug(ctx->log, "adding %p to tracking history fifo (ref = %d)\n",
                     ctx->tracking_history[0],
                     atomic_load(&ctx->tracking_history[0]->base.ref));

            if (ctx->n_tracking < TRACK_FRAMES)
                ctx->n_tracking++;

            gm_debug(ctx->log, "tracking history len = %d:", ctx->n_tracking);
            for (int i = 0; i < ctx->n_tracking; i++) {
                gm_debug(ctx->log, "%d) %p (ref = %d)", i,
                         ctx->tracking_history[i],
                         atomic_load(&ctx->tracking_history[i]->base.ref));
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

    gm_assert(tracking->ctx->log, atomic_load(&tracking->base.ref) == 0,
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

    gm_assert(tracking->ctx->log, atomic_load(&tracking->base.ref) >= 0,
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

    atomic_store(&tracking->base.ref, 0);
    tracking->base.api = &tracking->vtable;

    tracking->vtable.free = tracking_state_recycle;
    tracking->vtable.add_breadcrumb = tracking_add_breadcrumb;

    tracking->pool = pool;
    tracking->ctx = ctx;

    int labels_width = ctx->training_camera_intrinsics.width;
    int labels_height = ctx->training_camera_intrinsics.height;

    assert(labels_width);
    assert(labels_height);

    tracking->label_probs = (float *)xcalloc(labels_width *
                                             labels_height *
                                             ctx->n_labels, sizeof(float));

    tracking->skeleton.joints.resize(ctx->n_joints);
    tracking->skeleton.bones.clear();
    tracking->joints_processed = (float *)
      xcalloc(ctx->n_joints, 3 * sizeof(float));

    gm_assert(ctx->log, ctx->max_depth_pixels,
              "Undefined maximum number of depth pixels");

    tracking->depth = (float *)
      xcalloc(ctx->max_depth_pixels, sizeof(float));

    gm_assert(ctx->log, ctx->max_video_pixels,
              "Undefined maximum number of video pixels");

    tracking->stage_data.resize(N_TRACKING_STAGES, {});

#if 0
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
#endif

    return tracking;
}

static void
prediction_free(struct gm_mem_pool *pool,
                void *self,
                void *user_data)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)self;

    gm_assert(prediction->ctx->log, prediction->n_tracking == 0,
              "Freeing prediction that has tracking references");

    delete prediction;
}

static void
prediction_recycle(struct gm_prediction *self)
{
    struct gm_prediction_impl *prediction = (struct gm_prediction_impl *)self;
    struct gm_mem_pool *pool = prediction->pool;

    gm_assert(prediction->ctx->log, atomic_load(&prediction->base.ref) == 0,
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

    gm_assert(prediction->ctx->log, atomic_load(&prediction->base.ref) >= 0,
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

    atomic_store(&prediction->base.ref, 0);
    prediction->base.api = &prediction->vtable;

    prediction->vtable.free = prediction_recycle;
    prediction->vtable.add_breadcrumb = prediction_add_breadcrumb;

    prediction->pool = pool;

    prediction->ctx = ctx;
    prediction->n_tracking = 0;

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
             tracking, atomic_load(&tracking->base.ref),
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

#ifdef __linux__
    if (ret == 0) {
        pthread_setname_np(ctx->detect_thread, "Glimpse Track");
    }
#endif

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
    gm_context_clear_tracking(ctx);

    /* Free the prediction pool. The user must have made sure to unref any
     * predictions before destroying the context.
     */
    mem_pool_free(ctx->prediction_pool);

    /* Make sure all resourced are returned to their pools before destroying
     * the pools which will in-turn destroy the resources...
     */
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
        rdt_tree_destroy(ctx->decision_trees[i]);
    xfree(ctx->decision_trees);

    if (ctx->joint_params)
        jip_free(ctx->joint_params);

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

    pthread_mutex_init(&ctx->liveness_lock, NULL);
    pthread_mutex_init(&ctx->debug_viz_mutex, NULL);
    pthread_cond_init(&ctx->skel_track_cond, NULL);
    pthread_mutex_init(&ctx->skel_track_cond_mutex, NULL);
    pthread_mutex_init(&ctx->tracking_swap_mutex, NULL);
    pthread_mutex_init(&ctx->frame_ready_mutex, NULL);
    pthread_cond_init(&ctx->frame_ready_cond, NULL);

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
    int max_trees = 10;
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
                rdt_tree_load_from_buf(logger,
                                       (uint8_t *)gm_asset_get_buffer(tree_asset),
                                       gm_asset_get_length(tree_asset),
                                       &catch_err);
            if (!ctx->decision_trees[i]) {
                gm_warn(ctx->log,
                        "Failed to open binary decision tree '%s': %s",
                        name, catch_err);
                free(catch_err);
                catch_err = NULL;
            }
        } else {
            free(catch_err);
            catch_err = NULL;

            name = json_name;
            tree_asset = gm_asset_open(logger,
                                       json_name,
                                       GM_ASSET_MODE_BUFFER,
                                       &catch_err);
            if (!tree_asset) {
                if (ctx->n_decision_trees == 0) {
                    gm_warn(ctx->log,
                            "Failed to open JSON decision tree '%s': %s",
                            name, catch_err);
                }
                free(catch_err);
                break;
            }

            /* XXX: Technically we should pass a NUL terminated string but
             * since we're assuming we're passing a valid Json Object then we
             * can rely on parsing terminating on the closing '}' without
             * depending on finding a terminating NUL. Otherwise we would
             * have to copy the asset into a larger buffer so we can
             * explicitly add the NUL.
             */
            JSON_Value *js = json_parse_string((const char *)gm_asset_get_buffer(tree_asset));
            if (js) {
                ctx->decision_trees[i] =
                    rdt_tree_load_from_json(ctx->log, js,
                                            false, // don't load incomplete trees
                                            &catch_err);
                if (!ctx->decision_trees[i]) {
                    gm_warn(ctx->log,
                            "Failed to open JSON decision tree '%s': %s",
                            name, catch_err);
                    xfree(catch_err);
                    catch_err = NULL;
                }
            } else {
                gm_warn(ctx->log, "Failed to parse JSON decision tree '%s'\n",
                        name);
            }

        }

        gm_asset_close(tree_asset);

        if (!ctx->decision_trees[i]) {
            break;
        }

        gm_info(logger, "Opened decision tree '%s'", name);
        ctx->n_decision_trees++;
    }

    if (!ctx->n_decision_trees) {
        gm_throw(logger, err, "Failed to open any decision tree assets");
        gm_context_destroy(ctx);
        return NULL;
    } else {
        gm_info(logger, "Loaded %d decision trees", ctx->n_decision_trees);
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
    ctx->training_camera_intrinsics.width = labels_width;
    ctx->training_camera_intrinsics.height = labels_height;
    ctx->training_camera_intrinsics.cx = 86;
    ctx->training_camera_intrinsics.cy = 112;
    ctx->training_camera_intrinsics.fx = 217.461437772;
    ctx->training_camera_intrinsics.fy = 217.461437772;

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
            ctx->joint_params = jip_load_from_json(logger, root, err);
            json_value_free(root);
        }

        xfree(js_string);
        gm_asset_close(joint_params_asset);

        if (!ctx->joint_params) {
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


    ctx->motion_detection = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "motion_detection";
    prop.desc = "Enable motion-based human segmentation";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->motion_detection;
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

    ctx->skeleton_validation = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "skeleton_validation";
    prop.desc = "Whether to validate if inferred skeletons are likely to be "
                "human.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->skeleton_validation;
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
    prop.type = GM_PROPERTY_ENUM;
    prop.enum_state.ptr = &ctx->debug_label;

    struct gm_ui_enumerant enumerant;
    enumerant = gm_ui_enumerant();
    enumerant.name = "most likely";
    enumerant.desc = "Visualize Most Probable Labels";
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

    ctx->debug_cloud_mode = 0;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "cloud_mode";
    prop.desc = "debug mode for point cloud visualization";
    prop.type = GM_PROPERTY_ENUM;
    prop.enum_state.ptr = &ctx->debug_cloud_mode;

    enumerant = gm_ui_enumerant();
    enumerant.name = "none";
    enumerant.desc = "Don't create a debug point cloud";
    enumerant.val = 0;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "video";
    enumerant.desc = "Texture with video";
    enumerant.val = 1;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "depth";
    enumerant.desc = "Texture with depth";
    enumerant.val = 2;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "class";
    enumerant.desc = "Motion classification";
    enumerant.val = 3;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    prop.enum_state.n_enumerants = ctx->cloud_mode_enumerants.size();
    prop.enum_state.enumerants = ctx->cloud_mode_enumerants.data();
    ctx->properties.push_back(prop);

    /*
     * XXX: note we have to be careful with the initialization of stage
     * elements. Unlike the properties above, stages contain vectors so if we
     * were to use a local variable for temporary stage initialization (like
     * with the properties) then once the variable goes out of scope it will
     * result in the vector being destroyed.
     */


    ctx->stages.resize(N_TRACKING_STAGES);
    for (int i = 0; i < N_TRACKING_STAGES; i++) {
        ctx->stages[i].stage_id = TRACKING_STAGE_START;
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_START;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "acquire";
        stage.desc = "Captures a new frame to process for tracking";

        stage.images.push_back((struct image_generator)
                               {
                                   "video",
                                   "Video frame acquired from camera",
                                   gm_tracking_create_rgb_video,
                               });

        stage.images.push_back((struct image_generator)
                               {
                                   "depth",
                                   "Depth frame acquired from camera",
                                   gm_tracking_create_rgb_depth,
                               });

        ctx->apply_depth_distortion = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "apply_depth_distortion";
        prop.desc = "Apply the distortion model of depth camera";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->apply_depth_distortion;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_GAP_FILLED;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "gap_fill";
        stage.desc = "Fill gaps and apply min/max depth thresholding";

        ctx->min_depth = 0.5;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "min_depth";
        prop.desc = "throw away points nearer than this";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->min_depth;
        prop.float_state.min = 0.0;
        prop.float_state.max = 10;
        stage.properties.push_back(prop);

        ctx->max_depth = 3.5;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "max_depth";
        prop.desc = "throw away points further than this";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->max_depth;
        prop.float_state.min = 0.5;
        prop.float_state.max = 10;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_DOWNSAMPLED;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "downsample";
        stage.desc = "Downsamples the native-resolution depth data";

        ctx->seg_res = 1;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_res";
        prop.desc = "Resolution divider for running human segmentation";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->seg_res;
        prop.int_state.min = 1;
        prop.int_state.max = 4;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_GROUND_SPACE;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "ground_align";
        stage.desc = "Projects depth into ground-aligned space";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_SPACE;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_align";
        stage.desc = "Project into stable 'codebook' space for motion analysis";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_NAIVE_FLOOR;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "naive_find_floor";
        stage.desc = "Find floor to attempt naive single-person segmentation";

        ctx->floor_threshold = 0.1f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "floor_threshold";
        prop.desc = "The threshold from the lowest points of a potential person "
            "cluster to filter out when looking for the floor.";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->floor_threshold;
        prop.float_state.min = 0.01f;
        prop.float_state.max = 0.3f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_NAIVE_CLUSTER;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "naive_cluster";
        stage.desc = "Cluster based on assumptions about single-person tracking";

        ctx->cluster_tolerance = 0.10f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_tolerance";
        prop.desc = "Distance threshold when clustering points";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_tolerance;
        prop.float_state.min = 0.01f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->cluster_min_width = 0.15f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_min_width";
        prop.desc = "Minimum width of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_width;
        prop.float_state.min = 0.1f;
        prop.float_state.max = 1.0f;
        stage.properties.push_back(prop);

        ctx->cluster_min_height = 0.8f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_min_height";
        prop.desc = "Minimum height of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_height;
        prop.float_state.min = 0.1f;
        prop.float_state.max = 1.5f;
        stage.properties.push_back(prop);

        ctx->cluster_min_depth = 0.05f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_min_depth";
        prop.desc = "Minimum depth of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_depth;
        prop.float_state.min = 0.05f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->cluster_max_width = 2.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_max_width";
        prop.desc = "Maximum width of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_width;
        prop.float_state.min = 0.5f;
        prop.float_state.max = 3.0f;
        stage.properties.push_back(prop);

        ctx->cluster_max_height = 2.45f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_max_height";
        prop.desc = "Maximum height of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_height;
        prop.float_state.min = 1.0f;
        prop.float_state.max = 4.0f;
        stage.properties.push_back(prop);

        ctx->cluster_max_depth = 1.5f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "na_cluster_max_depth";
        prop.desc = "Maximum depth of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_depth;
        prop.float_state.min = 0.5f;
        prop.float_state.max = 3.0f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_CLASSIFY;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_classify";
        stage.desc = "Analyse and classify motion in codebook space for segmentation";
        stage.images.push_back((struct image_generator)
                               {
                                   "codebook_classifications",
                                   "Codebook classifications from motion analysis",
                                   gm_tracking_create_rgb_depth_classification,
                               });

        ctx->seg_tb = 0.05f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_tb";
        prop.desc = "Segmentation bucket threshold";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->seg_tb;
        prop.float_state.min = 0.001f;
        prop.float_state.max = 0.1f;
        stage.properties.push_back(prop);

        ctx->seg_tf = 0.2f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_tf";
        prop.desc = "Segmentation flickering threshold";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->seg_tf;
        prop.float_state.min = 0.05f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->seg_N = 100;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_N";
        prop.desc = "Segmentation max existing mean weight";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->seg_N;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->seg_b = 3;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_b";
        prop.desc = "Segmentation flickering frame consecutiveness threshold";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->seg_b;
        prop.int_state.min = 1;
        prop.int_state.max = 10;
        stage.properties.push_back(prop);

        ctx->seg_gamma = 100;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_gamma";
        prop.desc = "Segmentation max flickering frame occurence";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->seg_gamma;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->seg_alpha = 200;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_alpha";
        prop.desc = "Segmentation frame-time for uninteresting objects";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->seg_alpha;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->seg_psi = 0.8f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_psi";
        prop.desc = "Segmentation ratio for uninteresting object matches";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->seg_psi;
        prop.float_state.min = 0.f;
        prop.float_state.max = 1.f;
        stage.properties.push_back(prop);

        ctx->seg_timeout = 3.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "seg_timeout";
        prop.desc = "Unused segmentation codeword recycle timeout";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->seg_timeout;
        prop.float_state.min = 0.2f;
        prop.float_state.max = 10.f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_CLUSTER;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_cluster";
        stage.desc = "Cluster based on motion-based codebook classifications";
        stage.images.push_back((struct image_generator)
                               {
                                   "candidate_cluster",
                                   "All candidate clusters found, before selection",
                                   gm_tracking_create_rgb_candidate_clusters,
                               });

        ctx->cluster_tolerance = 0.10f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_tolerance";
        prop.desc = "Distance threshold when clustering points";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_tolerance;
        prop.float_state.min = 0.01f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->cluster_min_width = 0.15f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_min_width";
        prop.desc = "Minimum width of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_width;
        prop.float_state.min = 0.1f;
        prop.float_state.max = 1.0f;
        stage.properties.push_back(prop);

        ctx->cluster_min_height = 0.8f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_min_height";
        prop.desc = "Minimum height of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_height;
        prop.float_state.min = 0.1f;
        prop.float_state.max = 1.5f;
        stage.properties.push_back(prop);

        ctx->cluster_min_depth = 0.05f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_min_depth";
        prop.desc = "Minimum depth of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_min_depth;
        prop.float_state.min = 0.05f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->cluster_max_width = 2.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_max_width";
        prop.desc = "Maximum width of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_width;
        prop.float_state.min = 0.5f;
        prop.float_state.max = 3.0f;
        stage.properties.push_back(prop);

        ctx->cluster_max_height = 2.45f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_max_height";
        prop.desc = "Maximum height of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_height;
        prop.float_state.min = 1.0f;
        prop.float_state.max = 4.0f;
        stage.properties.push_back(prop);

        ctx->cluster_max_depth = 1.5f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cb_cluster_max_depth";
        prop.desc = "Maximum depth of a human cluster";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_max_depth;
        prop.float_state.min = 0.5f;
        prop.float_state.max = 3.0f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_FILTER_CLUSTERS;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "filter_clusters";
        stage.desc = "Filter plausible person clusters";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_PROJECT_CLUSTERS;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "project_clusters";
        stage.desc = "Project plausible person clusters into 2D depth buffers";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_SELECT_CANDIDATE_CLUSTER;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "select_cluster";
        stage.desc = "Select cluster to run label inferrence on (points before projection into depth image)";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_LABEL_INFERENCE;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "label_inference";
        stage.desc = "Infer per-pixel body part labels";
        stage.images.push_back((struct image_generator)
                               {
                                   "labels",
                                   "Inferred labels",
                                   gm_tracking_create_rgb_label_map,
                               });

        ctx->flip_labels = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "li_flip_labels";
        prop.desc = "Use horizontal image-flipping to enhance inference";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->flip_labels;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_JOINT_WEIGHTS;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "joint_weights";
        stage.desc = "Map body-part labels to per-joint weights";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_JOINT_INFERENCE;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "joint_inference";
        stage.desc = "Infer position of skeleton joints";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_REFINE_SKELETON;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "refine_skeleton";
        stage.desc = "Try and clean up issues with the inferred skeleton joints";

        ctx->bone_length_variance = 0.05f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "bone_length_variance";
        prop.desc = "Maximum allowed variance of bone length between frames";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->bone_length_variance;
        prop.float_state.min = 0.f;
        prop.float_state.max = 0.2f;
        stage.properties.push_back(prop);

        ctx->bone_rotation_variance = 360.f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "bone_rotation_variance";
        prop.desc = "Maximum allowed rotation of a bone (degrees/s)";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->bone_rotation_variance;
        prop.float_state.min = 100.f;
        prop.float_state.max = 1500.f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_SANITIZE_SKELETON;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "sanitize_skeleton";
        stage.desc = "Try and clean up issues with the inferred skeleton joints";

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
        stage.properties.push_back(prop);

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
        stage.properties.push_back(prop);

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
        stage.properties.push_back(prop);

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
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_UPDATE_CODEBOOK;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "update_codebook";
        stage.desc = "Update the codebook state ready for processing motion of future frames";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    for (int i = 1; i < N_TRACKING_STAGES; i++) {
        gm_assert(ctx->log, ctx->stages[i].stage_id != TRACKING_STAGE_START,
                  "Uninititialized stage description %d", i);
    }


    ctx->debug_pipeline_stage = TRACKING_STAGE_START;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "debug_stage";
    prop.desc = "tracking pipeline stage to inspect";
    prop.type = GM_PROPERTY_ENUM;
    prop.enum_state.ptr = &ctx->debug_pipeline_stage;

    for (int i = 0; i < N_TRACKING_STAGES; i++) {
        enumerant = gm_ui_enumerant();
        enumerant.name = ctx->stages[i].name;
        enumerant.desc = ctx->stages[i].desc;
        enumerant.val = i;
        ctx->cloud_stage_enumerants.push_back(enumerant);
    }
    prop.enum_state.n_enumerants = ctx->cloud_stage_enumerants.size();
    prop.enum_state.enumerants = ctx->cloud_stage_enumerants.data();
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    return ctx;
}

void
gm_context_set_config(struct gm_context *ctx, JSON_Value *json_config)
{
    gm_props_from_json(ctx->log,
                       gm_context_get_ui_properties(ctx),
                       json_config);

    JSON_Object *stages =
        json_object(json_object_get_value(json_object(json_config), "_stages"));
    int n_stages = gm_context_get_n_stages(ctx);
    for (int i = 0; i < n_stages; i++) {
        const char *stage_name = gm_context_get_stage_name(ctx, i);
        struct gm_ui_properties *stage_props =
            gm_context_get_stage_ui_properties(ctx, i);

        JSON_Value *json_stage_props =
            json_object_get_value(stages, stage_name);

        gm_props_from_json(ctx->log,
                           stage_props,
                           json_stage_props);
    }
}

void
gm_context_set_max_depth_pixels(struct gm_context *ctx, int max_pixels)
{
    ctx->max_depth_pixels = max_pixels;
}

void
gm_context_set_max_video_pixels(struct gm_context *ctx, int max_pixels)
{
    ctx->max_video_pixels = max_pixels;
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
    return &ctx->training_camera_intrinsics;
}

int
gm_context_get_n_stages(struct gm_context *ctx)
{
    return ctx->stages.size();
}

const char *
gm_context_get_stage_name(struct gm_context *ctx,
                          int stage)
{
    gm_assert(ctx->log, stage >=0 && stage < (int)ctx->stages.size(),
              "Out of range stage index");

    return ctx->stages[stage].name;
}

const char *
gm_context_get_stage_description(struct gm_context *ctx,
                                 int stage)
{
    gm_assert(ctx->log, stage >=0 && stage < (int)ctx->stages.size(),
              "Out of range stage index");

    return ctx->stages[stage].desc;
}

int
gm_context_get_stage_n_images(struct gm_context *ctx,
                              int stage_id)
{
    gm_assert(ctx->log, stage_id >=0 && stage_id < (int)ctx->stages.size(),
              "Out of range stage index (%d)", stage_id);
    struct gm_pipeline_stage &stage = ctx->stages[stage_id];

    return stage.images.size();
}

const char *
gm_context_get_stage_nth_image_name(struct gm_context *ctx,
                                    int stage_id,
                                    int n)
{
    gm_assert(ctx->log, stage_id >= 0 && stage_id < (int)ctx->stages.size(),
              "Out of range stage index (%d)", stage_id);
    struct gm_pipeline_stage &stage = ctx->stages[stage_id];

    gm_assert(ctx->log, n >=0 && n < (int)stage.images.size(),
              "Out of range stage %s image index (%d)", stage.name, n);

    return stage.images[n].name;
}

const char *
gm_context_get_stage_nth_image_description(struct gm_context *ctx,
                                           int stage_id,
                                           int n)
{
    gm_assert(ctx->log, stage_id >= 0 && stage_id < (int)ctx->stages.size(),
              "Out of range stage index (%d)", stage_id);
    struct gm_pipeline_stage &stage = ctx->stages[stage_id];

    gm_assert(ctx->log, n >=0 && n < (int)stage.images.size(),
              "Out of range stage %s image index (%d)", stage.name, n);

    return stage.images[n].desc;
}

struct gm_ui_properties *
gm_context_get_stage_ui_properties(struct gm_context *ctx, int stage)
{
    gm_assert(ctx->log, stage >=0 && stage < (int)ctx->stages.size(),
              "Out of range stage index");

    return &ctx->stages[stage].properties_state;
}

uint64_t
gm_context_get_average_stage_duration(struct gm_context *ctx,
                                      int stage)
{
    gm_assert(ctx->log, stage >=0 && stage < (int)ctx->stages.size(),
              "Out of range stage index");

    return 0; // FIXME
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

bool
gm_tracking_create_rgb_label_map(struct gm_tracking *_tracking,
                                 int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    uint8_t n_labels = ctx->n_labels;

    *width = (int)tracking->training_camera_intrinsics.width;
    *height = (int)tracking->training_camera_intrinsics.height;

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

    return true;
}

bool
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
#if 1
        struct color rgb = stops_color_from_val(ctx->depth_color_stops,
                                                ctx->n_depth_color_stops,
                                                ctx->depth_color_stops_range,
                                                depth);
        (*output)[off * 3] = rgb.r;
        (*output)[off * 3 + 1] = rgb.g;
        (*output)[off * 3 + 2] = rgb.b;
#else
        depth = std::max(ctx->min_depth, std::min(ctx->max_depth, depth));
        uint8_t shade = (uint8_t)
            ((depth - ctx->min_depth) /
             (ctx->max_depth - ctx->min_depth) * 255.f);
        (*output)[off * 3] = shade;
        (*output)[off * 3 + 1] = shade;
        (*output)[off * 3 + 2] = shade;
#endif
    }

    return true;
}

bool
gm_tracking_create_rgb_video(struct gm_tracking *_tracking,
                             int *width, int *height, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;
    struct gm_frame *frame = tracking->frame;

    *width = (int)frame->video_intrinsics.width;
    *height = (int)frame->video_intrinsics.height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    int rot_width = tracking->video_camera_intrinsics.width;
    enum gm_format format = frame->video_format;
    enum gm_rotation rotation = frame->camera_rotation;
    uint8_t *video = (uint8_t *)frame->video->data;

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
    case GM_FORMAT_BGR_U8:
        foreach_xy_off(*width, *height) {
            with_rotated_rx_ry_roff(x, y, *width, *height,
                                    rotation, rot_width,
                                    {
                                        (*output)[roff*3] = video[off*3+2];
                                        (*output)[roff*3+1] = video[off*3+1];
                                        (*output)[roff*3+2] = video[off*3];
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
    case GM_FORMAT_BGRX_U8:
    case GM_FORMAT_BGRA_U8:
        foreach_xy_off(*width, *height) {
            with_rotated_rx_ry_roff(x, y, *width, *height,
                                    rotation, rot_width,
                                    {
                                        (*output)[roff*3] = video[off*4+2];
                                        (*output)[roff*3+1] = video[off*4+1];
                                        (*output)[roff*3+2] = video[off*4];
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
        return false;
    }

    // Output is rotated, so make sure output width/height are correct
    if (rotation == GM_ROTATION_90 || rotation == GM_ROTATION_270) {
        std::swap(*width, *height);
    }

    return true;
}

bool
gm_tracking_create_rgb_candidate_clusters(struct gm_tracking *_tracking,
                                          int *width, int *height,
                                          uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    //struct gm_context *ctx = tracking->ctx;

    if (!tracking->cluster_labels) {
        return false;
    }

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

    return true;
}

bool
gm_tracking_create_rgb_depth_classification(struct gm_tracking *_tracking,
                                            int *width, int *height,
                                            uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    if (!tracking->depth_class) {
        return false;
    }

    *width = (int)tracking->depth_class->width;
    *height = (int)tracking->depth_class->height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        depth_classification_to_rgb((enum seg_class)tracking->depth_class->points[off].label,
                                    (*output) + off * 3);
    }

    return true;
}

const struct gm_point_rgba *
gm_tracking_get_debug_point_cloud(struct gm_tracking *_tracking,
                                  int *n_points)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    *n_points = tracking->debug_cloud.size();
    return (struct gm_point_rgba *)tracking->debug_cloud.data();
}

const struct gm_point_rgba *
gm_tracking_get_debug_lines(struct gm_tracking *_tracking,
                            int *n_lines)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    gm_assert(tracking->ctx->log,
              tracking->debug_lines.size() % 2 == 0,
              "Odd number of points in debug_lines array");
    *n_lines = tracking->debug_lines.size() / 2;
    return tracking->debug_lines.data();
}

uint64_t
gm_tracking_get_duration(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    return tracking->duration_ns;
}

uint64_t
gm_tracking_get_stage_duration(struct gm_tracking *_tracking,
                               int stage_index)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_index];

    return stage_data.duration_ns;
}

#if 0
const struct gm_point_rgba *
gm_tracking_get_stage_debug_point_cloud(struct gm_tracking *_tracking,
                                        int stage_index,
                                        int *n_points)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_index];

    return stage_data.debug_point_cloud.data();
}

const struct gm_point_rgba *
gm_tracking_get_stage_debug_lines(struct gm_tracking *_tracking,
                                  int stage_index,
                                  int *n_lines)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_index];

    return stage_data.debug_lines.data();
}
#endif

bool
gm_tracking_create_stage_rgb_image(struct gm_tracking *_tracking,
                                   int stage_index,
                                   int image_index,
                                   int *width,
                                   int *height,
                                   uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage &stage = ctx->stages[stage_index];

    gm_assert(ctx->log, image_index >=0 && image_index < (int)stage.images.size(),
              "Out of range stage %s image index (%d)", stage.name, image_index);

    if (stage.images[image_index].create_rgb_image)
        return stage.images[image_index].create_rgb_image(_tracking,
                                                          width, height,
                                                          output);
    else
        return false;
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

bool
gm_tracking_was_successful(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    return tracking->success;
}

struct gm_skeleton *
gm_skeleton_new(struct gm_context *ctx, struct gm_joint *joints,
                float confidence, float distance, uint64_t timestamp)
{
    struct gm_skeleton *skeleton = new struct gm_skeleton(ctx->n_joints);

    for (int j = 0; j < ctx->n_joints; ++j) {
        skeleton->joints[j] = joints[j];
    }

    skeleton->confidence = confidence;
    skeleton->distance = distance;
    skeleton->timestamp = timestamp;

    build_bones(ctx, *skeleton);

    return skeleton;
}

struct gm_skeleton *
gm_skeleton_new_from_json(struct gm_context *ctx,
                          const char *asset_name)
{
    char *catch_err = NULL;
    struct gm_asset *json_asset = gm_asset_open(ctx->log,
                                                asset_name,
                                                GM_ASSET_MODE_BUFFER,
                                                &catch_err);
    if (!json_asset) {
        gm_error(ctx->log,
                 "Failed to open skeleton json asset '%s': %s",
                 asset_name, catch_err);
        free(catch_err);
        return NULL;
    }

    const char *buffer = (const char *)gm_asset_get_buffer(json_asset);
    JSON_Value *js;
    if (!buffer || !(js = json_parse_string(buffer))) {
        gm_error(ctx->log,
                 "Failed to parse JSON asset '%s'", asset_name);
        gm_asset_close(json_asset);
        return NULL;
    }

    JSON_Array *bones = json_object_get_array(json_object(js), "bones");
    if (!bones) {
        gm_error(ctx->log,
                 "Failed to find bones in JSON asset '%s'", asset_name);
        json_value_free(js);
        gm_asset_close(json_asset);
        return NULL;
    }

    struct gm_joint joints[ctx->n_joints];
    memset(joints, 0, ctx->n_joints * sizeof(struct gm_joint));
    for (int j = 0; j < ctx->n_joints; ++j) {
        joints[j].name = joint_name(j);
        char *bone_name = strdup(joint_name(j));
        char *bone_part = strchr(bone_name, (int)'.');
        if (bone_part) {
            bone_part[0] = '\0';
            ++bone_part;

            bool found = false;
            for (size_t b = 0; b < json_array_get_count(bones); ++b) {
                JSON_Object *bone_obj = json_array_get_object(bones, b);
                if (strcmp(json_object_get_string(bone_obj, "name"),
                           bone_name) == 0) {
                    if (json_object_has_value(bone_obj, bone_part)) {
                        JSON_Array *joint_array =
                            json_object_get_array(bone_obj, bone_part);
                        joints[j].x = (float)
                            json_array_get_number(joint_array, 0);
                        joints[j].y = (float)
                            json_array_get_number(joint_array, 1);
                        joints[j].z = (float)
                            json_array_get_number(joint_array, 2);
                        joints[j].confidence = 1000.f;
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                gm_warn(ctx->log, "Joint '%s' not found in JSON asset '%s'",
                        bone_name, asset_name);
            }
        } else {
            gm_warn(ctx->log, "Can't derive bone name from joint name '%s'",
                    bone_name);
        }
        free(bone_name);
    }

    json_value_free(js);
    gm_asset_close(json_asset);

    return gm_skeleton_new(ctx, joints, 0, 0, 0);
}

void
displace_joints(const struct gm_context *ctx,
                struct gm_skeleton *skeleton,
                struct gm_bone &bone,
                float *displacement)
{
    skeleton->joints[bone.tail].x -= displacement[0];
    skeleton->joints[bone.tail].y -= displacement[1];
    skeleton->joints[bone.tail].z -= displacement[2];

    for (size_t b = 0; b < skeleton->bones.size(); ++b) {
        struct gm_bone &candidate_bone = skeleton->bones[b];
        if (candidate_bone.head == bone.tail) {
            displace_joints(ctx, skeleton, candidate_bone, displacement);
        }
    }
}

struct gm_skeleton *
gm_skeleton_resize(struct gm_context *ctx,
                   const struct gm_skeleton *skeleton,
                   const struct gm_skeleton *ref_skeleton,
                   int parent_joint)
{
    if (skeleton->bones.size() != ref_skeleton->bones.size()) {
        gm_error(ctx->log,
                 "Mismatching skeletons passed to gm_skeleton_resize()");
        return NULL;
    }

    struct gm_skeleton *resized = new struct gm_skeleton(skeleton);

    // Resize the bones then recalculate the joint positions based on the bones
    std::queue<size_t> leftover_bones;

    for (size_t b = 0; b < resized->bones.size(); ++b) {
        resized->bones[b].length = ref_skeleton->bones[b].length;
        if (resized->bones[b].head == parent_joint) {
            leftover_bones.push(b);
        }
    }

    while (!leftover_bones.empty()) {
        size_t b = leftover_bones.front();
        leftover_bones.pop();
        struct gm_bone &bone = resized->bones[b];

        // Recalculate tail position
        float length_mult = ref_skeleton->bones[b].length /
            skeleton->bones[b].length;
        float displacement[3];
        displacement[0] =
            resized->joints[bone.tail].x -
            (resized->joints[bone.head].x +
             ((skeleton->joints[bone.tail].x - skeleton->joints[bone.head].x) *
              length_mult));
        displacement[1] =
            resized->joints[bone.tail].y -
            (resized->joints[bone.head].y +
             ((skeleton->joints[bone.tail].y - skeleton->joints[bone.head].y) *
              length_mult));
        displacement[2] =
            resized->joints[bone.tail].z -
            (resized->joints[bone.head].z +
             ((skeleton->joints[bone.tail].z - skeleton->joints[bone.head].z) *
              length_mult));

        // Shift all the connected bones' tail joints
        displace_joints(ctx, resized, bone, displacement);

        // Place the next bones onto the list
        for (size_t b = 0; b < resized->bones.size(); ++b) {
            if (resized->bones[b].head == bone.tail) {
                leftover_bones.push(b);
            }
        }
    }

    return resized;
}

bool
gm_skeleton_save(const struct gm_skeleton *skeleton,
                 const char *filename)
{
    JSON_Value *root = json_value_init_object();
    JSON_Value *bones = json_value_init_array();
    json_object_set_value(json_object(root), "bones", bones);

    int n_joints = gm_skeleton_get_n_joints(skeleton);
    for (int i = 0; i < n_joints ; ++i) {
        const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);

        // Don't save a skeleton if a joint wasn't able to be inferred
        if (!joint || joint->name == NULL) {
            json_value_free(root);
            return false;
        }

        char *bone_name = strdup(joint->name);
        char *bone_part = strchr(bone_name, (int)'.');
        if (bone_part) {
            bone_part[0] = '\0';
            ++bone_part;

            // Find bone, or create one if this is the first encounter
            JSON_Value *bone = NULL;
            for (int c = 0;
                 c < json_array_get_count(json_array(bones)); ++c)
            {
                JSON_Value *bone_obj =
                    json_array_get_value(json_array(bones), c);
                if (strcmp(json_object_get_string(json_object(bone_obj),
                                                  "name"), bone_name) == 0)
                {
                    bone = bone_obj;
                    break;
                }
            }
            if (!bone) {
                bone = json_value_init_object();
                json_object_set_string(json_object(bone), "name", bone_name);
                json_array_append_value(json_array(bones), bone);
            }

            JSON_Value *joint_array = json_value_init_array();
            json_object_set_value(json_object(bone), bone_part, joint_array);
            json_array_append_number(json_array(joint_array), joint->x);
            json_array_append_number(json_array(joint_array), joint->y);
            json_array_append_number(json_array(joint_array), joint->z);
        }
        free(bone_name);
    }

    json_serialize_to_file_pretty(root, filename);

    json_value_free(root);

    return true;
}

void
gm_skeleton_free(struct gm_skeleton *skeleton)
{
    delete skeleton;
}

int
gm_skeleton_get_n_joints(const struct gm_skeleton *skeleton)
{
    return (int)skeleton->joints.size();
}

int
gm_skeleton_get_n_bones(const struct gm_skeleton *skeleton)
{
    return (int)skeleton->bones.size();
}

const struct gm_bone *
gm_skeleton_get_bone(const struct gm_skeleton *skeleton, int bone)
{
    return &skeleton->bones[bone];
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

float
gm_skeleton_compare_angle(const struct gm_skeleton *skel_a,
                          const struct gm_skeleton *skel_b,
                          const struct gm_bone *bone)
{
    return bone_angle_diff(bone, skel_a, skel_b);
}

float
gm_skeleton_angle_diff_cumulative(const struct gm_skeleton *skel_a,
                                  const struct gm_skeleton *skel_b)
{
    float cumulative_angle = 0.f;
    for (int b = 0; b < gm_skeleton_get_n_bones(skel_a); ++b) {
        cumulative_angle +=
            gm_skeleton_compare_angle(skel_a, skel_b,
                                      gm_skeleton_get_bone(skel_a, b));
    }

    return cumulative_angle;
}

int
gm_bone_get_head(const struct gm_bone *bone)
{
    return bone->head;
}

int
gm_bone_get_tail(const struct gm_bone *bone)
{
    return bone->tail;
}

float
gm_bone_get_length(const struct gm_bone *bone)
{
    return bone->length;
}

void
gm_bone_get_angle(const struct gm_bone *bone, float *out_xyzw)
{
    if (out_xyzw) {
        out_xyzw[0] = bone->angle.x;
        out_xyzw[1] = bone->angle.y;
        out_xyzw[2] = bone->angle.z;
        out_xyzw[3] = bone->angle.w;
    }
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

    gm_debug(ctx->log, "gm_context_notify_frame: frame = %p, depth=%p, video=%p (w=%d, h=%d)",
             frame, frame->depth, frame->video,
             frame->video_intrinsics.width,
             frame->video_intrinsics.height);

    gm_assert(ctx->log, !ctx->destroying,
              "Spurious notification during tracking context destruction");

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    gm_assert(ctx->log, !ctx->destroying, "Spurious frame notification during destruction");
    gm_assert(ctx->log, ctx->frame_ready != frame, "Notified of the same frame");
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
                 atomic_load(&ctx->latest_tracking->base.ref));
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
        prediction->tracking_history[closest_frame]->frame->timestamp;
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

        struct gm_tracking_impl *frame1 = prediction->tracking_history[h1];
        struct gm_tracking_impl *frame2 = prediction->tracking_history[h2];
        float t = (timestamp - frame2->frame->timestamp) /
                  (float)(frame1->frame->timestamp - frame2->frame->timestamp);

        // Use linear interpolation to place the parent bone(s). We'll use
        // the interpolated angles to place the rest of the bones.
        for (std::vector<struct gm_bone>::iterator it =
             closest_skeleton.bones.begin();
             it != closest_skeleton.bones.end(); ++it) {
            struct gm_bone &bone = *it;
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
        for (std::vector<struct gm_bone>::iterator it =
             closest_skeleton.bones.begin();
             it != closest_skeleton.bones.end(); ++it) {
            struct gm_bone &bone = *it;
            const struct gm_bone *parent_bone =
                find_bone(closest_skeleton.bones, -1, bone.head);
            if (!parent_bone) {
                continue;
            }

            const struct gm_bone *frame2_bone =
                find_bone(frame2->skeleton.bones, bone.head, bone.tail);
            if (!frame2_bone) {
                continue;
            }

            const struct gm_bone *frame1_bone =
                find_bone(frame1->skeleton.bones, bone.head, bone.tail);
            if (!frame1_bone) {
                continue;
            }

            // Find the angle to rotate the parent bone. Note, we're relying
            // on bones being stored in an order where we can rely on the
            // bone's parent being seen before any descendents.
            glm::mat3 rotate = glm::mat3_cast(
                glm::slerp(frame2_bone->angle,
                           frame1_bone->angle, t));

            glm::vec3 parent_vec = glm::normalize(
                glm::vec3(prediction->skeleton.joints[parent_bone->tail].x -
                          prediction->skeleton.joints[parent_bone->head].x,
                          prediction->skeleton.joints[parent_bone->tail].y -
                          prediction->skeleton.joints[parent_bone->head].y,
                          prediction->skeleton.joints[parent_bone->tail].z -
                          prediction->skeleton.joints[parent_bone->head].z));
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

