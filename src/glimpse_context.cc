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

/*
 * (Our cluster_codebook_classified_points() implementation was based
 *  on PCL's pcl::OrganizedConnectedComponentSegmentation::segment...)
 *
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
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
#include "joints_inferrer.h"
#include "rdt_tree.h"
#include "jip.h"
#include "image_utils.h"

#include "glimpse_log.h"
#include "glimpse_mem_pool.h"
#include "glimpse_assets.h"
#include "glimpse_data.h"
#include "glimpse_context.h"

#undef GM_LOG_CONTEXT
#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Tracking"
#else
#define GM_LOG_CONTEXT "ctx"
#endif

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)

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

enum debug_cloud_mode {
    DEBUG_CLOUD_MODE_NONE,
    DEBUG_CLOUD_MODE_VIDEO,
    DEBUG_CLOUD_MODE_DEPTH,
    DEBUG_CLOUD_MODE_CODEBOOK_LABELS,
    DEBUG_CLOUD_MODE_LABELS,
    DEBUG_CLOUD_MODE_EDGES,

    N_DEBUG_CLOUD_MODES
};

enum {
    CODEBOOK_DEBUG_VIEW_POINT_CLOUD,
    CODEBOOK_DEBUG_VIEW_CODEBOOK,
};

enum tracking_stage {
    TRACKING_STAGE_START,
    TRACKING_STAGE_NEAR_FAR_CULL_AND_INFILL,
    TRACKING_STAGE_DOWNSAMPLED,
    TRACKING_STAGE_EDGE_DETECT,
    TRACKING_STAGE_GROUND_SPACE,

    TRACKING_STAGE_NAIVE_FLOOR,
    TRACKING_STAGE_NAIVE_CLUSTER,

    TRACKING_STAGE_CODEBOOK_RETIRE_WORDS,
    TRACKING_STAGE_CODEBOOK_RESOLVE_BACKGROUND,
    TRACKING_STAGE_CODEBOOK_SPACE,
    TRACKING_STAGE_CODEBOOK_CLASSIFY,
    TRACKING_STAGE_CODEBOOK_CLUSTER,

    TRACKING_STAGE_FILTER_CLUSTERS,
    TRACKING_STAGE_CROP_CLUSTER_IMAGE,

    TRACKING_STAGE_SELECT_CANDIDATE_CLUSTER,
    TRACKING_STAGE_LABEL_INFERENCE,
    TRACKING_STAGE_JOINT_WEIGHTS,
    TRACKING_STAGE_JOINT_INFERENCE,
    TRACKING_STAGE_REFINE_SKELETON,

    TRACKING_STAGE_SANITIZE_SKELETON,

    TRACKING_STAGE_UPDATE_CODEBOOK,

    TRACKING_STAGE_UPDATE_HISTORY,

    N_TRACKING_STAGES
};

enum edge_detect_mode {
    EDGE_DETECT_MODE_NONE,
    EDGE_DETECT_MODE_X_ONLY,
    EDGE_DETECT_MODE_Y_ONLY,
    EDGE_DETECT_MODE_XY,
};

enum image_format {
    IMAGE_FORMAT_X8,
    IMAGE_FORMAT_XHALF,
    IMAGE_FORMAT_XFLOAT,
};

struct trail_crumb
{
    char tag[32];
    int n_frames;
    void *backtrace_frame_pointers[10];
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
    float mean;

    // The number of values ->mean is based on; for maintaining
    // a rolling average...
    int n;

    uint64_t create_timestamp;
    uint64_t create_frame_counter;
    uint64_t last_update_timestamp;
    uint64_t last_update_frame_count;

    int n_consecutive_update_runs;
};

// Depth pixel classification for segmentation
enum codebook_class
{
    CODEBOOK_CLASS_BACKGROUND,
    CODEBOOK_CLASS_FLAT,
    CODEBOOK_CLASS_FLICKERING,
    CODEBOOK_CLASS_FLAT_AND_FLICKERING,
    CODEBOOK_CLASS_FOREGROUND_OBJ_TO_IGNORE, // E.g. a bag / chair
    CODEBOOK_CLASS_FOREGROUND,
    CODEBOOK_CLASS_FAILED_CANDIDATE,
    CODEBOOK_CLASS_TRACKED,
};

struct candidate_cluster {
    unsigned label; // indexes into tracking->cluster_indices[]

    // 2D bounds (inclusive)
    int min_x_2d = INT_MAX;
    int max_x_2d = -1;
    int min_y_2d = INT_MAX;
    int max_y_2d = -1;

    float min_x = FLT_MAX;
    float max_x = -FLT_MAX;
    float min_y = FLT_MAX;
    float max_y = -FLT_MAX;
    float min_z = FLT_MAX;
    float max_z = -FLT_MAX;
};

struct joint_dist
{
    float min;
    float mean;
    float max;
};

#define MAX_BONE_CHILDREN 4

struct gm_bone_info
{
    int idx; // For indexing into ctx->bone_info[]

    int parent; // parent bone index
    int children[MAX_BONE_CHILDREN];
    int n_children;

    int head; // joint index
    int tail; // joint index

    float min_length;
    float mean_length;
    float max_length;

    bool has_rotation_constraint;
    glm::quat avg_rotation; // Average bone rotation
    float max_radius;       // Maximum variance from average rotation
};

struct gm_bone
{
    int idx; // For indexing into ctx->bone_info[]

    bool valid;

    // Cached distance between bone's head and tail joints
    // (see update_bones())
    float length;

    // XXX:
    //
    // This is the rotation between the parent bone's head->tail vector and
    // this bone's head->tail vector.
    //
    // I.e. it's _not_ a cumulative coordinate space transformation for
    // children of this bone.
    //
    glm::quat angle;
};

struct gm_skeleton {
    struct gm_context *ctx;
    std::vector<struct gm_joint> joints;
    std::vector<struct gm_bone> bones;
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

    /* So we can report average durations in UIs */
    uint64_t total_time_ns;
    int n_invocations;
    int n_frames; // average across whole frames (possibly multiple invocations)

    /* So we can report median filtered durations in UIs */
    std::vector<int> invocation_duration_hist;
    int invocation_duration_hist_head; // circular buffer ptr

    std::vector<int> frame_duration_hist;
    int frame_duration_hist_head; // circular buffer ptr

    std::vector<image_generator> images;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
};

/* A pipeline_stage maintains global information about a stage while
 * _stage_data can track information about a specific tracking run
 */
struct gm_pipeline_stage_data {

    // NB: a stage like label inference may be invoked multiple times
    // per-frame while tracking for different candidate clusters
    uint64_t frame_duration_ns; // total over frame
    std::vector<uint64_t> durations; // individual invocations

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

    /* If h2 == -1 then skeleton is a copy of tracking_history[h1]->skeleton
     * else skeleton is an interpolation of tracking_history[h1] and
     * tracking_history[2]
     *
     * These are just preserved for debugging purposes
     */
    int h1;
    int h2;
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

    /* XXX: these are currently a copy of ctx->basis_depth_to_video_extrinsics
     * and don't take into account device rotation
     */
    struct gm_extrinsics depth_to_video_extrinsics;
    bool extrinsics_set;

    /* Based on the depth_camera_intrinsics but taking into account the seg_res
     * resolution
     */
    struct gm_intrinsics downsampled_intrinsics;

    struct gm_frame *frame;

    // Depth data, in meters
    float *depth;

    // Label probability tables
    std::vector<float> label_probs;
    int label_probs_width;
    int label_probs_height;

    // The unprojected full-resolution depth cloud
    pcl::PointCloud<pcl::PointXYZL>::Ptr depth_cloud;

    // The depth cloud downsampled for segmentation
    pcl::PointCloud<pcl::PointXYZL>::Ptr downsampled_cloud;

    // naive or codebook segmentation/clustering
    std::vector<pcl::PointIndices> cluster_indices;

    // The ground-aligned segmentation-resolution depth cloud
    pcl::PointCloud<pcl::PointXYZL>::Ptr ground_cloud;

    std::vector<struct gm_point_rgba> debug_cloud;
    // It's useful to associate some intrinsics with the debug cloud to help
    // with visualizing it...
    struct gm_intrinsics debug_cloud_intrinsics;

    // While building the debug_cloud we sometimes track indices that map
    // back to some other internal cloud which may get used for colouring
    // the debug points.
    std::vector<int> debug_cloud_indices;

    std::vector<struct gm_point_rgba> debug_lines;

    std::vector<char *> debug_text;

    // Whether any person clouds were tracked in this frame
    bool success;

    // Whether the tracking was done based on a 'paused' camera frame
    bool paused;

    // Inferred joint positions
    InferredJoints *joints;
    struct gm_skeleton skeleton;
    struct gm_skeleton skeleton_corrected;

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
     * without destroying the context via stop_tracking_thread()
     */
    bool stopping;
    bool destroying;

    int max_depth_pixels;
    int max_video_pixels;

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

    pthread_t tracking_thread;
    dlib::frontal_face_detector detector;

    dlib::shape_predictor face_feature_detector;

    RDTree **decision_trees;
    int n_decision_trees;

    // Incremented for each tracking iteration
    uint64_t frame_counter;

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

    int codebook_debug_view;

    pthread_mutex_t skel_track_cond_mutex;
    pthread_cond_t skel_track_cond;

    struct gm_mem_pool *prediction_pool;

    int n_labels;
    JSON_Value *label_map;

    int n_joints;
    JSON_Value *joint_map;
    JIParams *joint_params;
    std::vector<const char *> joint_blender_names; // (pointers into joint_map data)
    std::vector<const char *> joint_names;
    std::vector<enum gm_joint_semantic> joint_semantics;
    struct joints_inferrer *joints_inferrer;

    int n_bones;
    std::vector<struct gm_bone_info> bone_info;

    bool apply_depth_distortion;

    float min_depth;
    float max_depth;
    bool clamp_to_max_depth;

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
    bool naive_seg_fallback;
    bool cluster_from_prev;
    float cluster_from_prev_dist_threshold;
    float cluster_from_prev_time_threshold;

    bool codebook_frozen;
    float codebook_foreground_scrub_timeout;
    uint64_t codebook_last_foreground_scrub_timestamp;
    float codebook_clear_timeout;
    uint64_t codebook_last_clear_timestamp;
    float codeword_timeout;
    float codebook_bg_threshold;
    float codebook_flat_threshold;
    float codebook_clear_tracked_threshold;
    float codebook_keep_back_most_threshold;
    pcl::PointCloud<pcl::Label>::Ptr codebook_cluster_labels_scratch;
    float codebook_cluster_tolerance;
    bool codebook_cluster_merge_large_neighbours;
    bool codebook_cluster_infill;
    int codebook_tiny_cluster_threshold;
    int codebook_large_cluster_threshold;
    int debug_codebook_cluster_idx;
    int codeword_mean_n_max;
    int codeword_flicker_max_run_len;
    int codeword_flicker_max_quiet_frames;
    int codeword_obj_min_n;
    float codeword_obj_max_frame_to_n_ratio;
    int debug_codebook_layer;

    std::vector<float> inference_cluster_depth_image;
    std::vector<float> inference_cluster_weights;
    bool use_threads;
    bool flip_labels;

    bool fast_clustering;

    bool joint_refinement;
    float max_joint_refinement_delta;

    float bone_length_variance;
    float bone_rotation_variance;

    bool prediction_dampen_large_deltas;
    bool prediction_interpolate_angles;
    float max_prediction_delta;
    float prediction_decay;

    float sanitisation_window;
    bool joint_velocity_sanitisation;
    bool bone_length_sanitisation;
    bool bone_rotation_sanitisation;
    bool use_bone_map_annotation;

    float joint_velocity_threshold;
    float joint_outlier_factor;
    float bone_length_outlier_factor;
    float bone_rotation_outlier_factor;

    bool skeleton_validation;
    float skeleton_min_confidence;
    float skeleton_max_distance;

    bool debug_predictions;
    float debug_prediction_offset;

    int n_depth_color_stops;
    float depth_color_stops_range;
    struct color_stop *depth_color_stops;

    int n_heat_color_stops;
    float heat_color_stops_range;
    struct color_stop *heat_color_stops;

    std::vector<struct gm_ui_enumerant> cloud_stage_enumerants;
    std::vector<struct gm_ui_enumerant> cloud_mode_enumerants;
    std::vector<struct gm_ui_enumerant> codebook_debug_view_enumerants;
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

    /* A re-usable allocation for label probabilities that might
     * get swapped into the latest tracking object
     */
    std::vector<float> label_probs_back;

    /* When paused we're careful to not preserve any results for tracking
     * frames so the future frames will be processed with the same initial
     * state.
     */
    bool paused;

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
    int n_tracking;
    uint64_t last_tracking_success_timestamp;

    /* Whether we succeed or fail to track a frame we store the resulting
     * tracking object here after processing a frame. latest_tracking->success
     * should be checked to know if we detected a person.
     * This is what _get_latest_tracking() will return, unless
     * latest_paused_frame is non-NULL (see below).
     */
    struct gm_tracking_impl *latest_tracking;

    /* Tracking objects resulting from processing paused frames will never
     * update ctx->latest_tracking, since tracking itself may refer to
     * ctx->latest_tracking and while paused we need to be able to
     * repeatedly process the same frame consistently if no properties are
     * changed. At the end of tracking we instead save paused tracking
     * objects here and _get_latest_tracking() will return this if
     * not NULL.
     */
    struct gm_tracking_impl *latest_paused_tracking;

    /* Note: we need to be careful about updating this codebook state when
     * processing paused frames to have a deterministic initial state for
     * each iteration of the same paused frame...
     */
    struct gm_pose codebook_pose;
    glm::mat4 start_to_codebook;
    std::vector<std::vector<struct seg_codeword>> seg_codebook;
    uint64_t last_codebook_update_time;
    uint64_t last_codebook_update_frame_counter;

    /* If we're processing a paused frame then we will start by making a full
     * copy of the codebook into here so that we can make temporary updates
     * which will be discarded at the end so as to avoid changing the ctx
     * state seen for the next iteration of the same paused frame.
     */
    std::vector<std::vector<struct seg_codeword>> pause_frame_seg_codebook;

    /* This vector is only used for temporary state during the motion based
     * segmentation stage but we hang the vector off the context to avoid
     * having to repeatedly allocate a buffer for each tracking iteration
     */
    std::vector<struct seg_codeword *> seg_codebook_bg;

    /* Note: this lock covers the aggregated metrics under ctx->stages[] too */
    pthread_mutex_t aggregate_metrics_mutex;
    int n_frames;
    uint64_t total_tracking_duration;

    int edge_detect_mode;
    std::vector<struct gm_ui_enumerant> edge_detect_mode_enumerants;
    float edge_threshold;
    bool delete_edges;
    int edge_break_x;
    int edge_break_y;

    // Note: vector<bool> is implemented as an array of bits as a special case...
    std::vector<bool> edge_detect_scratch;
};

struct PointCmp {
    int x;
    int y;
    int lx;
    int ly;
};

/* As a general rule anything that's needed from the ctx for tracking that
 * might need modifying will be copied over to this scratch_state object and
 * becomes the authority for that state for tracking. At the end of tracking we
 * will only copy modifications back to the context for non-'paused' frames.
 *
 * This design lets us use the glimpse_viewer to pause playback of a recording
 * and while we are paused the same frame can be repeatedly processed with the
 * same starting state each time so we can understand the effect of property
 * changes including the motion based segmentation that wont behave as if all
 * movement has stopped.
 */
struct pipeline_scratch_state
{
    bool paused;

    int debug_pipeline_stage;
    int debug_cloud_mode;

    uint64_t frame_counter;

    int seg_res;

    bool done_edge_detect;

    /* The reason we copy the tracking history here at the start of tracking is
     * that we might have a frame that's marked as a discontinuity which should
     * result in us clearing the tracking history, but as noted above we need
     * to avoid any ctx state changes for paused frames.
     */
    //struct gm_tracking_impl *tracking_history[TRACK_FRAMES];
    //int n_tracking;

    bool to_start_valid;
    glm::mat4 to_start;
    bool to_ground_valid;
    glm::mat4 to_ground;

    struct gm_pose codebook_pose;
    glm::mat4 start_to_codebook;
    std::vector<std::vector<struct seg_codeword>> *seg_codebook;

    bool codebook_frozen;

    // true after updating point labels with motion-based classification
    bool codebook_classified;

    // naive segmentation
    int naive_fx;
    int naive_fy;
    std::vector<bool> done_mask;
    std::queue<struct PointCmp> flood_fill;
    float naive_floor_y;

    std::vector<candidate_cluster> candidate_clusters;
    std::vector<candidate_cluster> person_clusters;
    int current_person_cluster;
    int best_person_cluster;

    // per-cluster inference
    bool done_label_inference;
    InferredJoints *joints_candidate;

    float confidence;
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


/* simple hash to cheaply randomize how we will gaps in our data without
 * too much bias
 */
static uint32_t
xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
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

static inline float
distance_between(const float *point1, const float *point2)
{
    return sqrtf(powf(point1[0] - point2[0], 2.f) +
                 powf(point1[1] - point2[1], 2.f) +
                 powf(point1[2] - point2[2], 2.f));
}

static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// The overall intention here is that this transform should be used to limit
// how far into the past/future predictions should be made relative a base
// timestamp, considering that predictions become less and less accurate the
// further forwards/backwards we project.
//
// This function will leave timestamps unchanged so long as the delta from
// the base timestamp is below max_delta_ms.
//
// After the max_delta_ms threshold we have a second delta threshold
// at max_delta_ms + max_decay_ms, where max_decay_ms is:
//
//   max_decay_ms = max_delta_ms * delta_decay
//
// For timestamps that have a delta from the base timestamp of delta_ms that's
// between max_delta_ms and max_decay_ms this function maps the timestamp to:
//
//   base_ms + max_delta_ms + sqrt((delta_ms/max_decay_ms) * max_decay_ms)
//
// Note: the delta_ms value is clamped to never be > max_decay_ms
// Note: the largest effective delta is `max_delta_ms + sqrt(max_decay_ms)`
// Note: the implementation is actually done using nanosecond units
//
static uint64_t
calculate_decayed_timestamp(uint64_t base,
                            uint64_t timestamp,
                            float max_delta_ms, // XXX: milliseconds
                            float delta_decay)
{
    uint64_t max_delta_ns = (uint64_t)(max_delta_ms * 1000000.0);
    uint64_t delta_ns = (timestamp < base) ? base - timestamp : timestamp - base;

    if (delta_ns <= max_delta_ns)
        return timestamp;

    // After the max_delta_ns threshold we have a second delta threshold
    // at max_delta_ns + max_decay_ns...
    //
    uint64_t max_decay_ns = (uint64_t)((double)max_delta_ns * delta_decay);

    // How much have we overshot max_delta_ns?
    //
    // Note: Deltas that reach the second max_decay_ns threshold are clamped...
    //
    uint64_t decay_time_ns = std::min(delta_ns - max_delta_ns, max_decay_ns);

    // Map deltas > max_delta_ns to a value in the range from max_delta_ns to
    // (max_delta_ns + sqrt(max_decay_ns)) based on the function:
    //
    //   sqrt((decay/max_decay) * max_decay)
    //
    // NB: (decay/max_decay) is clamped to the range [0:1]
    //
    uint64_t decay_ns = (uint64_t)(sqrt(decay_time_ns / (double)max_decay_ns) *
                                   max_decay_ns);

    return (timestamp < base) ?
        (base - max_delta_ns) - decay_ns :
        base + max_delta_ns + decay_ns;
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

static inline bool
is_bone_length_diff(const struct gm_bone &ref_bone,
                    const struct gm_bone &bone,
                    float max_variance)
{
    return fabsf(bone.length - ref_bone.length) > max_variance;
}

static inline float
bone_angle_diff(struct gm_context *ctx,
                struct gm_bone *bone,
                struct gm_skeleton *ref_skel,
                struct gm_skeleton *skel)
{
    struct gm_bone_info &bone_info = ctx->bone_info[bone->idx];
    struct gm_joint &ref_head = ref_skel->joints[bone_info.head];
    struct gm_joint &ref_tail = ref_skel->joints[bone_info.tail];
    struct gm_joint &head = skel->joints[bone_info.head];
    struct gm_joint &tail = skel->joints[bone_info.tail];

    glm::vec3 ref_vec = glm::vec3(ref_tail.x - ref_head.x,
                                  ref_tail.y - ref_head.y,
                                  ref_tail.z - ref_head.z);
    glm::vec3 bone_vec = glm::vec3(tail.x - head.x,
                                   tail.y - head.y,
                                   tail.z - head.z);

    float angle = glm::degrees(acosf(
        glm::dot(glm::normalize(bone_vec), glm::normalize(ref_vec))));
    while (angle > 180.f)
        angle -= 360.f;
    return std::isnan(angle) ? 0.f : angle;
}

static inline bool
is_bone_angle_diff(struct gm_context *ctx,
                   struct gm_bone &bone,
                   struct gm_skeleton &ref_skel,
                   struct gm_skeleton &skel,
                   float time_delta,
                   float max_angle)
{
    float angle = bone_angle_diff(ctx, &bone, &ref_skel, &skel);
    /*float time = skel.timestamp > ref_skel.timestamp ?
        (float)((skel.timestamp - ref_skel.timestamp) / 1e9) :
        (float)((ref_skel.timestamp - skel.timestamp) / 1e9);*/
    float angle_delta = fabsf(angle) / time_delta;

    return angle_delta > max_angle;
}

static float
calc_cumulative_joint_difference(struct gm_context *ctx,
                                 struct gm_skeleton &skel,
                                 struct gm_skeleton &ref)
{
    float diff = 0.f;

    for (int i = 0; i < ctx->n_joints; ++i) {
        struct gm_joint &joint = skel.joints[i];
        struct gm_joint &ref_joint = ref.joints[i];

        if (joint.valid != ref_joint.valid) {
            continue;
        }

        diff += distance_between(&joint.x, &ref_joint.x);
    }

    return diff;
}

static int
calc_mismatched_bones(struct gm_context *ctx,
                      struct gm_skeleton &skel,
                      struct gm_skeleton &ref,
                      float time_delta)
{
    int n_bones = ctx->n_bones;

    gm_assert(ctx->log, skel.bones.size() == n_bones,
              "Skeleton doesn't have expected %d bones", n_bones);
    gm_assert(ctx->log, ref.bones.size() == n_bones,
              "Reference skeleton doesn't have expected %d bones", n_bones);

    int violations = 0;
    for (int i = 0; i < n_bones; ++i) {
        struct gm_bone &bone = skel.bones[i];
        struct gm_bone &ref_bone = ref.bones[i];
        struct gm_bone_info &bone_info = ctx->bone_info[i];

        // XXX: it's not entirely obvious what the best way of considering
        // bone validity is here but one aim is to avoid having a skeleton
        // where all bones are invalid looking better than any valid skeleton
        // so we count invalid bones as violations.
        if (!bone.valid)
            violations++;

        // We can't compare lengths or angles if either bone is invalid...
        if (!bone.valid || !ref_bone.valid)
            continue;

        if (is_bone_length_diff(bone, ref_bone, ctx->bone_length_variance))
            violations++;

        // Don't check for angle missmatches for the root bone...
        if (bone_info.parent >= 0) {
            if (is_bone_angle_diff(ctx, bone, ref, skel, time_delta,
                                   ctx->bone_rotation_variance))
            {
                violations++;
            }
        }
    }

    return violations;
}

static void
update_bones(struct gm_context *ctx, struct gm_skeleton &skeleton)
{
    int n_bones = ctx->n_bones;

    skeleton.bones.resize(n_bones);

    for (int i = 0; i < n_bones; i++) {
        struct gm_bone_info &bone_info = ctx->bone_info[i];
        struct gm_bone &bone = skeleton.bones[i];
        bone.idx = i;
        bone.valid = false;

        struct gm_joint &head = skeleton.joints[bone_info.head];
        struct gm_joint &tail = skeleton.joints[bone_info.tail];
        if (!head.valid || !tail.valid) {
            bone.length = 0;
            bone.angle = glm::quat();
            continue;
        }

        bone.length = distance_between(&tail.x, &head.x);

        int bone_parent_idx = bone_info.parent;
        if (bone_parent_idx >= 0) {
            struct gm_bone_info &parent_info = ctx->bone_info[bone_parent_idx];
            struct gm_joint &parent_head = skeleton.joints[parent_info.head];
            struct gm_joint &parent_tail = skeleton.joints[parent_info.tail];

            if (!parent_head.valid || !parent_tail.valid) {
                bone.length = 0;
                bone.angle = glm::quat();
                continue;
            }

            glm::vec3 bone_vec = glm::normalize(
                glm::vec3(tail.x - head.x,
                          tail.y - head.y,
                          tail.z - head.z));

            gm_assert(ctx->log,
                      (parent_tail.x == head.x &&
                       parent_tail.y == head.y &&
                       parent_tail.z == head.z),
                      "Expected bone[%d] head to == parent bone[%d] tail: head=(%f,%f,%f), parent tail=(%f,%f,%f)",
                      i, bone_parent_idx,
                      head.x, head.y, head.z,
                      parent_tail.x, parent_tail.y, parent_tail.z);

            glm::vec3 parent_vec = glm::normalize(
                glm::vec3(parent_tail.x - parent_head.x,
                          parent_tail.y - parent_head.y,
                          parent_tail.z - parent_head.z));

            glm::vec3 axis = glm::normalize(glm::cross(bone_vec, parent_vec));
            float angle = acosf(glm::dot(bone_vec, parent_vec));

            bone.angle = glm::angleAxis(angle, axis);

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

        bone.valid = true;
    }
}

static float
calc_skeleton_distance(struct gm_context *ctx,
                       struct gm_skeleton *skeleton)
{
    int n_bones = ctx->n_bones;
    float distance = 0.f;

    for (int b = 0; b < n_bones; ++b) {
        struct gm_bone_info &bone_info = ctx->bone_info[b];
        struct gm_bone &bone = skeleton->bones[b];

        if (!bone.valid) {
            continue;
        }

        float length = bone.length;
        if (length < bone_info.min_length) {
            distance += powf(bone_info.min_length - length, 2.f);
        } else if (length > bone_info.max_length) {
            distance += powf(length - bone_info.max_length, 2.f);
        }
    }
    return distance;
}

static void
copy_inferred_joints_to_skel_except(struct gm_skeleton &dest,
                                    InferredJoints *inferred_src,
                                    int except_joint_no)
{
    struct gm_context *ctx = dest.ctx;

    for (int i = 0; i < ctx->n_joints; i++) {
        if (i == except_joint_no)
            continue;

        /* Inference gives us a list of candidate clusters for each joint and
         * we're only considering the first candidate (with the highest
         * confidence)
         */
        LList *inferred_joint_list = inferred_src->joints[i];
        if (inferred_joint_list) {
            Joint *inferred = (Joint *)inferred_joint_list->data;
            dest.joints[i].valid = true;
            dest.joints[i].x = inferred->x;
            dest.joints[i].y = inferred->y;
            dest.joints[i].z = inferred->z;
        } else {
            dest.joints[i].valid = false;
            dest.joints[i].x = 0;
            dest.joints[i].y = 0;
            dest.joints[i].z = 0;
        }
    }
}

static void
refine_skeleton(struct gm_tracking_impl *tracking)
{
    struct gm_context *ctx = tracking->ctx;
    uint64_t time_threshold = tracking->frame->timestamp -
        (uint64_t)((double)ctx->max_joint_refinement_delta * 1e9);

    if (!ctx->joint_refinement || !ctx->n_tracking ||
        ctx->tracking_history[0]->frame->timestamp < time_threshold) {
        return;
    }

    float joint_diff =
        calc_cumulative_joint_difference(ctx, tracking->skeleton_corrected,
                                         ctx->tracking_history[0]->
                                         skeleton_corrected);

    // For each joint, we look at the cumulative distance between joints using
    // each joint cluster and if the distance is lower, we replace that joint
    // and continue.
    for (int j = 0; j < ctx->n_joints; ++j) {
        if (!tracking->joints->joints[j] ||
            !tracking->joints->joints[j]->next) {
            continue;
        }

        for (LList *l = tracking->joints->joints[j]->next; l; l = l->next) {
            struct gm_skeleton candidate_skeleton = {};

            candidate_skeleton.ctx = ctx;
            candidate_skeleton.joints.resize(ctx->n_joints);

            Joint *joint = (Joint *)l->data;
            candidate_skeleton.joints[j].valid = true;
            candidate_skeleton.joints[j].x = joint->x;
            candidate_skeleton.joints[j].y = joint->y;
            candidate_skeleton.joints[j].z = joint->z;

            copy_inferred_joints_to_skel_except(candidate_skeleton, // dest
                                                tracking->joints, // src
                                                j); // Don't overwrite this joint
            update_bones(ctx, candidate_skeleton);

            float cand_joint_diff =
                calc_cumulative_joint_difference(ctx, candidate_skeleton,
                                                 ctx->tracking_history[0]->
                                                 skeleton_corrected);
            if (cand_joint_diff <= joint_diff) {
                std::swap(tracking->skeleton_corrected, candidate_skeleton);
                joint_diff = cand_joint_diff;
            }
        }
    }
}

static void
interpolate_joints(struct gm_joint &a, struct gm_joint &b, float t,
                   struct gm_joint &out)
{
    out.x = a.x + (b.x - a.x) * t;
    out.y = a.y + (b.y - a.y) * t;
    out.z = a.z + (b.z - a.z) * t;
}

static bool
is_bone_length_valid(struct gm_context *ctx,
                     struct gm_bone &bone,
                     float avg_bone_length)
{
    if (ctx->use_bone_map_annotation) {
        struct gm_bone_info &bone_info = ctx->bone_info[bone.idx];
        if (bone.length < bone_info.min_length ||
            bone.length > bone_info.max_length) {
            return false;
        }
    }
    float bone_length_factor = (bone.length > avg_bone_length) ?
        bone.length / avg_bone_length : avg_bone_length / bone.length;

    return bone_length_factor <= ctx->bone_length_outlier_factor;
}

static bool
is_bone_rotation_valid(struct gm_context *ctx,
                       struct gm_bone &bone)
{
    if (ctx->use_bone_map_annotation) {
        struct gm_bone_info &bone_info = ctx->bone_info[bone.idx];
        if (bone_info.has_rotation_constraint) {
            float diff = glm::angle(bone.angle *
                                    glm::inverse(bone_info.avg_rotation));
            while (diff > M_PI) diff -= 2 * M_PI;
            if (fabsf(diff) > bone_info.max_radius) {
                return false;
            }
        }
    }

    return true;
}

static void
sanitise_joint_velocities(struct gm_context *ctx,
                          struct gm_skeleton &skeleton,
                          uint64_t timestamp,
                          uint64_t time_threshold)
{
    if (!ctx->joint_velocity_sanitisation) {
        return;
    }

    // We process the skeleton with regards to the 'parent' joint. At the
    // parent, we just have absolute position to look at, so we make sure it
    // hasn't moved too far too quickly and use the last good position if it
    // has.
    struct gm_tracking_impl **prev = ctx->tracking_history;
    bool changed = false;

    for (int b = 0; b <= ctx->n_bones; ++b) {
        // When b == 0, look at the bone head joint, otherwise look at the tail
        // of bone b - 1.
        // TODO: We assume bone 0 is the parent bone
        int joint;
        struct gm_bone_info *bone;
        if (b == 0) {
            bone = &ctx->bone_info[0];
            joint = bone->head;
        } else {
            bone = &ctx->bone_info[b - 1];
            joint = bone->tail;
        }

        // Find the last valid joint in history
        struct gm_joint &parent_joint = skeleton.joints[joint];
        struct gm_joint *prev_joint = NULL;
        int n_prev = -1;
        for (int i = 0; i < ctx->n_tracking; ++i) {
            if (prev[i]->skeleton_corrected.joints[joint].valid) {
                prev_joint = &prev[i]->skeleton_corrected.joints[joint];
                n_prev = i;
                break;
            }
        }

        // If there are no previous valid joints, we can't do any sanitisation
        if (!prev_joint) {
            continue;
        }

        // Calculate the current joint velocity and if it falls below a
        // particular threshold, don't bother with sanitisation
        float time = (float)((timestamp - prev[n_prev]->frame->timestamp) / 1e9);
        float velocity = distance_between(&parent_joint.x,
                                          &prev_joint->x) / time;

        if (velocity < ctx->joint_velocity_threshold) {
            continue;
        }

        // Check the velocity of the joint at each previous tracking frame
        float velocities[ctx->n_tracking - 1];
        float avg_velocity = velocity;
        int n_velocities = 1;
        for (int j = 0; j < ctx->n_tracking - 1; ++j) {
            struct gm_joint &joint1 = prev[j]->skeleton.joints[joint];
            struct gm_joint &joint2 = prev[j+1]->skeleton.joints[joint];
            if (!joint1.valid || !joint2.valid ||
                prev[j+1]->frame->timestamp < time_threshold)
            {
                velocities[j] = FLT_MAX;
                continue;
            }

            time = (float)((prev[j]->frame->timestamp -
                            prev[j+1]->frame->timestamp) / 1e9);
            velocities[j] = distance_between(&joint1.x,
                                             &joint2.x) / time;
            avg_velocity += velocities[j];
            ++n_velocities;
        }
        avg_velocity /= n_velocities;

        // If this new joint is an outlier, use the last non-outlier position
        float outlier_threshold = avg_velocity * ctx->joint_outlier_factor;
        if (n_velocities > 1 && velocity > outlier_threshold) {
            for (int j = 0; j < ctx->n_tracking - 1; ++j) {
                if (velocities[j] <= outlier_threshold) {
                    gm_debug(ctx->log, "Bone joint (%s) average velocity: "
                             "%.2f, correction: %.2f -> %.2f",
                             ctx->joint_names[joint],
                             avg_velocity, velocity, velocities[j]);
                    struct gm_joint &prev_joint =
                        prev[j]->skeleton.joints[joint];

                    bool canUseDisplacement = false;
                    if (b != 0) {
                        // If this isn't the parent joint, find the bone that
                        // has it as a tail and use the displacement instead of
                        // the absolute position.
                        if (skeleton.joints[bone->head].valid &&
                            prev[j]->skeleton.joints[bone->head].valid) {
                            canUseDisplacement = true;
                        }
                    }

                    if (canUseDisplacement) {
                        struct gm_joint &head_joint =
                            skeleton.joints[bone->head];
                        struct gm_joint &prev_head_joint =
                            prev[j]->skeleton.joints[bone->head];
                        parent_joint.x =
                            head_joint.x + (prev_joint.x - prev_head_joint.x);
                        parent_joint.y =
                            head_joint.y + (prev_joint.y - prev_head_joint.y);
                        parent_joint.z =
                            head_joint.z + (prev_joint.z - prev_head_joint.z);
                    } else {
                        // If we can't correct the joint position wrt its
                        // displacement from a head joint, use the absolute
                        // position.
                        parent_joint = prev_joint;
                    }

                    parent_joint.valid = true;
                    changed = true;
                    break;
                }
            }
        }
    }

    // Update the bone metadata
    if (changed) {
        update_bones(ctx, skeleton);
    }
}

static void
sanitise_bone_lengths(struct gm_context *ctx,
                      struct gm_skeleton &skeleton,
                      uint64_t time_threshold)
{
    if (!ctx->bone_length_sanitisation) {
        return;
    }

    struct gm_tracking_impl **prev = ctx->tracking_history;

    for (auto &bone : skeleton.bones) {
        struct gm_bone_info &bone_info = ctx->bone_info[bone.idx];
        struct gm_joint &head = skeleton.joints[bone_info.head];
        struct gm_joint &tail = skeleton.joints[bone_info.tail];

        // Look at the length of each bone and the average length of that
        // bone in tracking history. If it deviates too far from the mean,
        // use the last non-outlier length.

        // Find the average length for this bone and see if this new length
        // conforms.
        float avg_bone_length = bone.length;
        int n_lengths = 1;
        for (int i = 0; i < ctx->n_tracking; ++i) {
            if (prev[i]->frame->timestamp < time_threshold) {
                break;
            }
            const struct gm_bone &prev_bone =
                prev[i]->skeleton.bones[bone.idx];
            if (ctx->use_bone_map_annotation) {
                struct gm_bone_info &bone_info = ctx->bone_info[bone.idx];
                if (bone.length < bone_info.min_length ||
                    bone.length > bone_info.max_length) {
                    continue;
                }
            }
            avg_bone_length += prev_bone.length;
            ++n_lengths;
        }
        avg_bone_length /= n_lengths;

        if (!is_bone_length_valid(ctx, bone, avg_bone_length)) {
            for (int i = 0; i < ctx->n_tracking; ++i) {
                struct gm_bone &prev_bone =
                    prev[i]->skeleton.bones[bone.idx];
                if (!is_bone_length_valid(ctx, prev_bone, avg_bone_length)) {
                    continue;
                }

                // Modify the tail position of this bone so that it's the
                // same length as the same bone in tracking history that
                // we've deemed to be valid.
                float new_length = prev_bone.length;

                gm_debug(ctx->log,
                         "Bone (%s->%s) average length: %.2f, "
                         "correction: %.2f -> %.2f",
                         ctx->joint_names[bone_info.head],
                         ctx->joint_names[bone_info.tail],
                         avg_bone_length, bone.length, new_length);

                glm::vec3 new_tail =
                    glm::vec3(head.x, head.y, head.z) +
                    (glm::normalize(
                         glm::vec3(tail.x - head.x,
                                   tail.y - head.y,
                                   tail.z - head.z)) * new_length);

                tail.x = new_tail.x;
                tail.y = new_tail.y;
                tail.z = new_tail.z;

                // Refresh bone info now the joint has changed
                update_bones(ctx, skeleton);
                break;
            }
        }
    }
}

static void
sanitise_bone_rotations(struct gm_context *ctx,
                        struct gm_skeleton &skeleton,
                        uint64_t time_threshold)
{
    if (!ctx->bone_rotation_sanitisation) {
        return;
    }

    struct gm_tracking_impl **prev = ctx->tracking_history;

    for (auto &bone : skeleton.bones) {
        struct gm_bone_info &bone_info = ctx->bone_info[bone.idx];

        // If this bone has no parent bone, we can't correct its angle
        int parent_bone_idx = bone_info.parent;
        if (parent_bone_idx < 0)
            continue;

        struct gm_bone_info &parent_bone_info = ctx->bone_info[parent_bone_idx];

        // Look at the change of the angle of rotation of each bone in
        // tracking history and compare the average of this to the current
        // change. If it exceeds it by too much, use the last rotation that
        // doesn't exceed this value.

        // Record which bones were derived from valid joints
        // If bone-map-annotation is being used, record bones as invalid if
        // their rotation doesn't lie within the annotated rotation constraints.
        bool bone_validity[ctx->n_tracking];
        for (int i = 0; i < ctx->n_tracking; ++i) {
            if (!prev[i]->skeleton.joints[bone_info.head].valid ||
                !prev[i]->skeleton.joints[bone_info.tail].valid ||
                prev[i]->frame->timestamp < time_threshold)
            {
                bone_validity[i] = false;
            } else {
                struct gm_bone &prev_bone = prev[i]->skeleton.bones[bone.idx];
                bone_validity[i] = is_bone_rotation_valid(ctx, prev_bone);
            }
        }

        // Find the average rotation change magnitude
        float bone_rots[ctx->n_tracking - 1];
        float bone_rot = fabsf(bone_angle_diff(ctx,
                                               &bone,
                                               &skeleton,
                                               &prev[0]->skeleton));
        float avg_bone_rot = bone_rot;
        int n_rots = 1;
        for (int i = 0; i < ctx->n_tracking - 1; ++i) {
            if (!bone_validity[i] || !bone_validity[i + 1])
            {
              bone_rots[i] = 180.f;
              continue;
            }

            bone_rots[i] = fabsf(bone_angle_diff(ctx,
                                                 &bone,
                                                 &prev[i]->skeleton,
                                                 &prev[i+1]->skeleton));
            gm_assert(ctx->log, !std::isnan(bone_rots[i]),
                      "Bone (%s->%s) angle diff is NaN "
                      "(%.2f, %.2f, %.2f->%.2f, %.2f, %.2f) v "
                      "(%.2f, %.2f, %.2f->%.2f, %.2f, %.2f)",
                      ctx->joint_names[bone_info.head],
                      ctx->joint_names[bone_info.tail],
                      prev[i]->skeleton.joints[bone_info.head].x,
                      prev[i]->skeleton.joints[bone_info.head].y,
                      prev[i]->skeleton.joints[bone_info.head].z,
                      prev[i]->skeleton.joints[bone_info.tail].x,
                      prev[i]->skeleton.joints[bone_info.tail].y,
                      prev[i]->skeleton.joints[bone_info.tail].z,
                      prev[i+1]->skeleton.joints[bone_info.head].x,
                      prev[i+1]->skeleton.joints[bone_info.head].y,
                      prev[i+1]->skeleton.joints[bone_info.head].z,
                      prev[i+1]->skeleton.joints[bone_info.tail].x,
                      prev[i+1]->skeleton.joints[bone_info.tail].y,
                      prev[i+1]->skeleton.joints[bone_info.tail].z);
            avg_bone_rot += bone_rots[i];
            ++n_rots;
        }
        avg_bone_rot /= n_rots;

        gm_debug(ctx->log, "Bone (%s->%s) average rot-mag: %.2f",
                 ctx->joint_names[bone_info.head],
                 ctx->joint_names[bone_info.tail],
                 avg_bone_rot);

        float bone_rot_factor = avg_bone_rot *
            ctx->bone_rotation_outlier_factor;
        if (bone_rot > bone_rot_factor || !is_bone_rotation_valid(ctx, bone)) {
            for (int i = 0; i < ctx->n_tracking; ++i) {
                if (bone_rots[i] > bone_rot_factor || !bone_validity[i]) {
                    continue;
                }

                gm_debug(ctx->log, "Bone (%s->%s) average rot-mag: %.2f, "
                         "correction: %.2f -> %.2f",
                         ctx->joint_names[bone_info.head],
                         ctx->joint_names[bone_info.tail],
                         avg_bone_rot, bone_rot, bone_rots[i]);

                const struct gm_bone *abs_prev_bone =
                    &ctx->tracking_history[i]->skeleton.bones[bone.idx];

                glm::mat3 rotate = glm::mat3_cast(abs_prev_bone->angle);

                glm::vec3 parent_vec = glm::normalize(
                    glm::vec3(skeleton.joints[parent_bone_info.tail].x -
                              skeleton.joints[parent_bone_info.head].x,
                              skeleton.joints[parent_bone_info.tail].y -
                              skeleton.joints[parent_bone_info.head].y,
                              skeleton.joints[parent_bone_info.tail].z -
                              skeleton.joints[parent_bone_info.head].z));

                glm::vec3 new_tail = ((parent_vec * rotate) * bone.length);
                new_tail.x += skeleton.joints[bone_info.head].x;
                new_tail.y += skeleton.joints[bone_info.head].y;
                new_tail.z += skeleton.joints[bone_info.head].z;

                skeleton.joints[bone_info.tail].x = new_tail.x;
                skeleton.joints[bone_info.tail].y = new_tail.y;
                skeleton.joints[bone_info.tail].z = new_tail.z;

                // Refresh bone info now the joint has changed
                update_bones(ctx, skeleton);
                break;
            }
        }
    }
}

static void
sanitise_skeleton(struct gm_context *ctx,
                  struct gm_skeleton &skeleton,
                  uint64_t timestamp)
{
    // We can't do any sanitisation without any tracking history
    if (!ctx->n_tracking) {
        return;
    }

    // We can't do any sanitisation if the last tracking history item is too old
    uint64_t time_threshold =
        timestamp - (uint64_t)((double)ctx->sanitisation_window * 1e9);
    if (ctx->tracking_history[0]->frame->timestamp < time_threshold) {
        return;
    }

    sanitise_joint_velocities(ctx, skeleton, timestamp, time_threshold);
    sanitise_bone_lengths(ctx, skeleton, time_threshold);
    sanitise_bone_rotations(ctx, skeleton, time_threshold);
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
        if ((input_->points[idx1].label == CODEBOOK_CLASS_FLICKERING ||
             input_->points[idx1].label == CODEBOOK_CLASS_FOREGROUND) &&
            (input_->points[idx2].label == CODEBOOK_CLASS_FLICKERING ||
             input_->points[idx2].label == CODEBOOK_CLASS_FOREGROUND)) {
            return fabsf(input_->points[idx1].z - input_->points[idx2].z) <
                depth_threshold_;
        }

        return false;
    }

  protected:
    float depth_threshold_;
};

/* TODO: combine the to_start and to_codebook matrices
 */
static inline void
project_point(float *point,
              struct gm_intrinsics *intrinsics,
              float *out_x,
              float *out_y)
{
    *out_x = ((point[0] * intrinsics->fx / point[2]) + intrinsics->cx);
    *out_y = ((point[1] * intrinsics->fy / point[2]) + intrinsics->cy);
}

static int
project_point_into_codebook(pcl::PointXYZL *point,
                            glm::mat4 to_start,
                            glm::mat4 start_to_codebook,
                            struct gm_intrinsics *intrinsics)
{
    if (std::isnan(point->z))
        return -1;

    glm::vec4 pt(point->x, point->y, point->z, 1.f);
    pt = (to_start * pt);
    pt = (start_to_codebook * pt);

    float transformed_pt[3] = { pt.x, pt.y, pt.z };
    point->x = pt.x;
    point->y = pt.y;
    point->z = pt.z;

    float nx, ny;
    project_point(&transformed_pt[0], intrinsics, &nx, &ny);

    int dnx = nx;
    int dny = ny;

    if (dnx < 0 || dnx >= intrinsics->width ||
        dny < 0 || dny >= intrinsics->height)
    {
        return -1;
    }

    return intrinsics->width * dny + dnx;
}

static inline bool
compare_point_depths(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
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

    return (d1 - d2) <= tolerance;
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

static void
tracking_state_free(struct gm_mem_pool *pool,
                    void *self,
                    void *user_data)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)self;
    struct gm_context *ctx = tracking->ctx;

    free(tracking->depth);

    free(tracking->face_detect_buf);

    if (tracking->frame) {
        gm_frame_unref(tracking->frame);
    }

    if (tracking->joints) {
        joints_inferrer_free_joints(ctx->joints_inferrer,
                                    tracking->joints);
    }

    delete tracking;
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
tracking_state_recycle(struct gm_tracking *self)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)self;
    struct gm_context *ctx = tracking->ctx;
    struct gm_mem_pool *pool = tracking->pool;

    gm_assert(ctx->log, atomic_load(&tracking->base.ref) == 0,
              "Unbalanced tracking unref");

    gm_frame_unref(tracking->frame);
    tracking->frame = NULL;

    if (tracking->joints) {
        joints_inferrer_free_joints(ctx->joints_inferrer,
                                    tracking->joints);
        tracking->joints = NULL;
    }

    for (auto string : tracking->debug_text) {
        free(string);
    }
    tracking->debug_text.resize(0);

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

    tracking->joints = NULL;

    tracking->skeleton.ctx = ctx;
    tracking->skeleton.joints.resize(ctx->n_joints);
    tracking->skeleton.bones.clear();

    gm_assert(ctx->log, ctx->max_depth_pixels,
              "Undefined maximum number of depth pixels");

    tracking->depth = (float *)
      xcalloc(ctx->max_depth_pixels, sizeof(float));

    tracking->label_probs.resize(ctx->max_depth_pixels);

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
label_probs_to_rgb(struct gm_context *ctx,
                   float *label_probs,
                   int n_labels,
                   uint8_t *rgb_out)
{
    if (ctx->debug_label == -1) {
        uint8_t label = 0;
        float pr = -1.0;

        for (int l = 0; l < n_labels; l++) {
            if (label_probs[l] > pr) {
                label = l;
                pr = label_probs[l];
            }
        }

        rgb_out[0] = default_palette[label].red;
        rgb_out[1] = default_palette[label].green;
        rgb_out[2] = default_palette[label].blue;
    } else {
        struct color col = stops_color_from_val(ctx->heat_color_stops,
                                                ctx->n_heat_color_stops,
                                                1,
                                                label_probs[ctx->debug_label]);
        rgb_out[0] = col.r;
        rgb_out[1] = col.g;
        rgb_out[2] = col.b;
    }
}

static bool
tracking_create_rgb_label_map(struct gm_tracking *_tracking,
                              int *width_out, int *height_out, uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    uint8_t n_labels = ctx->n_labels;

    int width = tracking->label_probs_width;
    int height = tracking->label_probs_height;

    if (tracking->label_probs.size() != width * height * n_labels)
        return false;

    *width_out = width;
    *height_out = height;

    if (!(*output)) {
        *output = (uint8_t *)malloc(width * height * 3);
    }

    gm_assert(ctx->log, ctx->debug_label < n_labels,
              "Can't create RGB map of invalid label %u",
              ctx->debug_label);

    foreach_xy_off(width, height) {
        float *label_probs = &tracking->label_probs[off * n_labels];

        uint8_t rgb[3];
        label_probs_to_rgb(ctx, label_probs, n_labels, rgb);

        (*output)[off * 3] = rgb[0];
        (*output)[off * 3 + 1] = rgb[1];
        (*output)[off * 3 + 2] = rgb[2];
    }

    return true;
}

static bool
tracking_create_rgb_depth(struct gm_tracking *_tracking,
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

static bool
tracking_create_rgb_video(struct gm_tracking *_tracking,
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

static bool
tracking_create_rgb_candidate_clusters(struct gm_tracking *_tracking,
                                       int *width_out, int *height_out,
                                       uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud = tracking->downsampled_cloud;
    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;

    if (!tracking->cluster_indices.size()) {
        return false;
    }

    int width = pcl_cloud->width;
    int height = pcl_cloud->height;
    *width_out = width;
    *height_out = height;

    if (!width || !height)
        return false;

    size_t out_size = width * height * 3;
    if (!(*output))
        *output = (uint8_t *)malloc(out_size);

    memset(*output, 0, out_size);

    int nth_large = -1;
    for (unsigned label = 0; label < cluster_indices.size(); label++) {
        auto &cluster = cluster_indices[label];

        if (cluster.indices.size() < ctx->codebook_large_cluster_threshold)
            continue;

        nth_large++;

        if (ctx->debug_codebook_cluster_idx != -1 &&
            ctx->debug_codebook_cluster_idx != nth_large)
        {
            continue;
        }

        for (int i : cluster.indices) {
            int x = i % width;
            int y = i / width;
            int off = width * y + x;

            if (x >= width || y >= height)
                continue;

            png_color *color =
                &default_palette[label % ARRAY_LEN(default_palette)];
            float shade = 1.f - (float)(label / ARRAY_LEN(default_palette)) / 10.f;
            (*output)[off * 3] = (uint8_t)(color->red * shade);
            (*output)[off * 3 + 1] = (uint8_t)(color->green * shade);
            (*output)[off * 3 + 2] = (uint8_t)(color->blue * shade);
        }
    }

    return true;
}

static void
depth_classification_to_rgb(enum codebook_class label, uint8_t *rgb_out)
{
    switch(label) {
    case CODEBOOK_CLASS_BACKGROUND:
        // Red
        rgb_out[0] = 0xff;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0x00;
        break;
    case CODEBOOK_CLASS_FLAT:
        // Dark Green
        rgb_out[0] = 0x00;
        rgb_out[1] = 0x80;
        rgb_out[2] = 0x00;
        break;
    case CODEBOOK_CLASS_FLICKERING:
        // Cyan
        rgb_out[0] = 0x00;
        rgb_out[1] = 0xff;
        rgb_out[2] = 0xff;
        break;
    case CODEBOOK_CLASS_FLAT_AND_FLICKERING:
        // Orange
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xA0;
        rgb_out[2] = 0x00;
        break;
    case CODEBOOK_CLASS_FOREGROUND_OBJ_TO_IGNORE:
        // Blue
        rgb_out[0] = 0x00;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0xFF;
        break;
    case CODEBOOK_CLASS_FOREGROUND:
        // White
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0xFF;
        break;
    case CODEBOOK_CLASS_FAILED_CANDIDATE:
        // Yellow
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0x00;
        break;
    case CODEBOOK_CLASS_TRACKED:
        // Green
        rgb_out[0] = 0x00;
        rgb_out[1] = 0xFF;
        rgb_out[2] = 0x00;
        break;
    case -1:
        // Invalid/unhandled value
        // Pink / Peach
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0x80;
        rgb_out[2] = 0x80;
        break;

    default:
        // unhandled value
        // Magenta
        rgb_out[0] = 0xFF;
        rgb_out[1] = 0x00;
        rgb_out[2] = 0xFF;
        break;
    }
}

static bool
tracking_create_rgb_depth_classification(struct gm_tracking *_tracking,
                                         int *width, int *height,
                                         uint8_t **output)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;

    if (!tracking->downsampled_cloud) {
        return false;
    }

    *width = (int)tracking->downsampled_cloud->width;
    *height = (int)tracking->downsampled_cloud->height;

    if (!(*output)) {
        *output = (uint8_t *)malloc((*width) * (*height) * 3);
    }

    foreach_xy_off(*width, *height) {
        depth_classification_to_rgb((enum codebook_class)tracking->downsampled_cloud->points[off].label,
                                    (*output) + off * 3);
    }

    return true;
}

static void
tracking_draw_line(struct gm_tracking_impl *tracking,
                   float x0, float y0, float z0,
                   float x1, float y1, float z1,
                   uint32_t rgba)
{
    struct gm_point_rgba p0 = { x0, y0, z0, rgba };
    struct gm_point_rgba p1 = { x1, y1, z1, rgba };
    tracking->debug_lines.push_back(p0);
    tracking->debug_lines.push_back(p1);
}

static void
tracking_draw_transformed_line(struct gm_tracking_impl *tracking,
                               float *start,
                               float *end,
                               uint32_t color,
                               glm::mat4 transform)
{
    glm::vec4 start_vec4(start[0], start[1], start[2], 1.f);
    glm::vec4 end_vec4(end[0], end[1], end[2], 1.f);

    start_vec4 = (transform * start_vec4);
    end_vec4 = (transform * end_vec4);

    tracking_draw_line(tracking,
                       start_vec4.x, start_vec4.y, start_vec4.z,
                       end_vec4.x, end_vec4.y, end_vec4.z,
                       color);
}

static void
tracking_draw_transformed_crosshair(struct gm_tracking_impl *tracking,
                                    float *pos,
                                    float size,
                                    uint32_t color,
                                    glm::mat4 transform)
{
    float half_size = size / 2.f;
    glm::vec4 axis_origin(pos[0], pos[1], pos[2], 1.f);
    glm::vec4 axis_x0(pos[0] - half_size, pos[1], pos[2], 1.f);
    glm::vec4 axis_x1(pos[0] + half_size, pos[1], pos[2], 1.f);
    glm::vec4 axis_y0(pos[0], pos[1] - half_size, pos[2], 1.f);
    glm::vec4 axis_y1(pos[0], pos[1] + half_size, pos[2], 1.f);
    glm::vec4 axis_z0(pos[0], pos[1], pos[2] - half_size, 1.f);
    glm::vec4 axis_z1(pos[0], pos[1], pos[2] + half_size, 1.f);

    axis_x0 = (transform * axis_x0);
    axis_x1 = (transform * axis_x1);
    axis_y0 = (transform * axis_y0);
    axis_y1 = (transform * axis_y1);
    axis_z0 = (transform * axis_z0);
    axis_z1 = (transform * axis_z1);

    tracking_draw_line(tracking,
                       axis_x0.x, axis_x0.y, axis_x0.z,
                       axis_x1.x, axis_x1.y, axis_x1.z,
                       color);
    tracking_draw_line(tracking,
                       axis_y0.x, axis_y0.y, axis_y0.z,
                       axis_y1.x, axis_y1.y, axis_y1.z,
                       color);
    tracking_draw_line(tracking,
                       axis_z0.x, axis_z0.y, axis_z0.z,
                       axis_z1.x, axis_z1.y, axis_z1.z,
                       color);
}

static void
tracking_draw_transformed_axis(struct gm_tracking_impl *tracking,
                               float *pos,
                               uint32_t *colors,
                               glm::mat4 transform)
{
    glm::vec4 axis_origin(pos[0], pos[1], pos[2], 1.f);
    glm::vec4 axis_x(pos[0] + 1.f, pos[1], pos[2], 1.f);
    glm::vec4 axis_y(pos[0], pos[1] + 1.f, pos[2], 1.f);
    glm::vec4 axis_z(pos[0], pos[1], pos[2] + 1.f, 1.f);

    axis_origin = (transform * axis_origin);
    axis_x = (transform * axis_x);
    axis_y = (transform * axis_y);
    axis_z = (transform * axis_z);

    tracking_draw_line(tracking,
                       axis_origin.x, axis_origin.y, axis_origin.z,
                       axis_x.x, axis_x.y, axis_x.z,
                       colors[0]);
    tracking_draw_line(tracking,
                       axis_origin.x, axis_origin.y, axis_origin.z,
                       axis_y.x, axis_y.y, axis_y.z,
                       colors[1]);
    tracking_draw_line(tracking,
                       axis_origin.x, axis_origin.y, axis_origin.z,
                       axis_z.x, axis_z.y, axis_z.z,
                       colors[2]);
}

static void
tracking_draw_transformed_grid(struct gm_tracking_impl *tracking,
                               float *center,
                               float full_size,
                               float cell_size,
                               uint32_t color,
                               glm::mat4 transform)

{
    float half_full_size = full_size / 2.0f;
    float corner0[] = { center[0] - half_full_size, center[1], center[2] - half_full_size };
    float corner1[] = { center[0] - half_full_size, center[1], center[2] + half_full_size };
    float corner2[] = { center[0] + half_full_size, center[1], center[2] + half_full_size };
    float corner3[] = { center[0] + half_full_size, center[1], center[2] - half_full_size };

    tracking_draw_transformed_line(tracking, corner0, corner1, color, transform);
    tracking_draw_transformed_line(tracking, corner1, corner2, color, transform);
    tracking_draw_transformed_line(tracking, corner2, corner3, color, transform);
    tracking_draw_transformed_line(tracking, corner3, corner0, color, transform);

    for (float off = cell_size; off < full_size; off += cell_size) {
        float end0[] = { corner0[0], corner0[1], corner0[2] + off };
        float end1[] = { corner3[0], corner3[1], corner3[2] + off };

        tracking_draw_transformed_line(tracking, end0, end1, color, transform);
    }
    for (float off = cell_size; off < full_size; off += cell_size) {
        float end0[] = { corner0[0] + off, corner0[1], corner0[2] };
        float end1[] = { corner1[0] + off, corner1[1], corner1[2] };

        tracking_draw_transformed_line(tracking, end0, end1, color, transform);
    }
}

static void
tracking_add_debug_text(struct gm_tracking_impl *tracking,
                        const char *fmt,
                        ...)
{
    va_list args;
    char *debug_text = NULL;
    va_start (args, fmt);
    vasprintf(&debug_text, fmt, args);
    va_end(args);
    if (debug_text) {
        tracking->debug_text.push_back(debug_text);
        gm_debug(tracking->ctx->log, "Tracking debug: %s", debug_text);
    }
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
    return tracking->success ? &tracking->skeleton_corrected : NULL;
}

const struct gm_skeleton *
gm_tracking_get_raw_skeleton(struct gm_tracking *_tracking)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    return tracking->success ? &tracking->skeleton : NULL;
}

const struct gm_point_rgba *
gm_tracking_get_debug_point_cloud(struct gm_tracking *_tracking,
                                  int *n_points,
                                  struct gm_intrinsics *debug_cloud_intrinsics)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    *n_points = tracking->debug_cloud.size();
    *debug_cloud_intrinsics = tracking->debug_cloud_intrinsics;
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

const char **
gm_tracking_get_debug_text(struct gm_tracking *_tracking,
                           int *n_strings)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    *n_strings = tracking->debug_text.size();
    return (const char **)tracking->debug_text.data();
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

    return stage_data.frame_duration_ns;
}

uint64_t
gm_tracking_get_stage_run_duration_avg(struct gm_tracking *_tracking,
                                       int stage_index)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_index];

    if (stage_data.frame_duration_ns)
        return stage_data.frame_duration_ns / stage_data.durations.size();
    else
        return 0;
}

uint64_t
gm_tracking_get_stage_run_duration_median(struct gm_tracking *_tracking,
                                          int stage_index)
{
    struct gm_tracking_impl *tracking = (struct gm_tracking_impl *)_tracking;
    struct gm_context *ctx = tracking->ctx;

    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage_data &stage_data = tracking->stage_data[stage_index];

    if (stage_data.durations.size() <= 1)
        return stage_data.frame_duration_ns;

    int len = stage_data.durations.size();
    uint64_t tmp[len];
    for (int i = 0; i < len; i++)
        tmp[i] = stage_data.durations[i];
    std::sort(tmp, tmp + len);

    return tmp[len/2];
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

uint64_t
gm_prediction_get_timestamp(struct gm_prediction *_prediction)
{
    struct gm_prediction_impl *prediction =
        (struct gm_prediction_impl *)_prediction;
    return prediction->timestamp;
}

struct gm_skeleton *
gm_prediction_get_skeleton(struct gm_prediction *_prediction)
{
    struct gm_prediction_impl *prediction =
        (struct gm_prediction_impl *)_prediction;
    return &prediction->skeleton;
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

static void
pcl_xyzl_cloud_from_buf_with_near_far_cull_and_infill(struct gm_context *ctx,
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
    bool clamp_max = ctx->clamp_to_max_depth;

    // May 'continue;' after setting a points position to NaN values
    // so expected to be use with a loop over a row of points...
#define near_far_cull_point_within_loop(point, off) \
    ({ \
        if (!std::isnormal(point.z) || \
            point.z < z_min) \
        { \
            point.x = point.y = point.z = nan; \
            point.label = -1; \
            pcl_cloud->points[off] = point; \
            continue; \
        } \
        if (point.z > z_max) \
        { \
            if (clamp_max) { \
                point.z = z_max; \
            } else { \
                point.x = point.y = point.z = nan; \
                point.label = -1; \
                pcl_cloud->points[off] = point; \
                continue; \
            } \
        } \
        point.x = (x - cx) * point.z * inv_fx; \
        point.y = -((y - cy) * point.z * inv_fy); \
        point.label = -1; \
        pcl_cloud->points[off] = point; \
    })

#define copy_row(Y) do { \
    int y = Y; \
    int row = y * width; \
    for (int x = 0; x < width; x++) { \
        int off = row + x; \
        pcl::PointXYZL point; \
        point.z = depth[off]; \
        near_far_cull_point_within_loop(point, off); \
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

            near_far_cull_point_within_loop(point, off);
        }
    }

    copy_row(height - 1);
#undef copy_row
#undef near_far_cull_point_within_loop
}

static void
add_debug_cloud_xyz_from_pcl_xyzl(struct gm_context *ctx,
                                  struct gm_tracking_impl *tracking,
                                  pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    gm_assert(ctx->log, debug_cloud.size() == debug_cloud_indices.size(),
              "Can't mix and match use of debug cloud indexing");

    for (int i = 0; i < (int)pcl_cloud->size(); i++) {
        struct gm_point_rgba point;

        point.x = pcl_cloud->points[i].x;
        point.y = pcl_cloud->points[i].y;
        point.z = pcl_cloud->points[i].z;
        point.rgba = 0xffffffff;

        debug_cloud.push_back(point);
        debug_cloud_indices.push_back(i);
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

    gm_assert(ctx->log, debug_cloud.size() == debug_cloud_indices.size(),
              "Can't mix and match use of debug cloud indexing");

    for (int i = 0; i < (int)pcl_cloud->size(); i++) {
        struct gm_point_rgba point;

        glm::vec4 pt(pcl_cloud->points[i].x,
                     pcl_cloud->points[i].y,
                     pcl_cloud->points[i].z,
                     1.f);
        pt = (transform * pt);

        point.x = pt.x;
        point.y = pt.y;
        point.z = pt.z;
        point.rgba = 0xffffffff;

        debug_cloud.push_back(point);
        debug_cloud_indices.push_back(i);
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

    gm_assert(ctx->log, debug_cloud.size() == debug_cloud_indices.size(),
              "Can't mix and match use of debug cloud indexing");

    int n_points = pcl_cloud->points.size();

    for (int i : indices) {
        struct gm_point_rgba point;

        point.x = pcl_cloud->points[i].x;
        point.y = pcl_cloud->points[i].y;
        point.z = pcl_cloud->points[i].z;
        point.rgba = 0xffffffff;

        debug_cloud.push_back(point);
        debug_cloud_indices.push_back(i);

        gm_assert(ctx->log, i < n_points, "Out-of-bounds index (%d > n_points=%d)",
                  i, n_points);
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
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    gm_assert(ctx->log, debug_cloud_indices.size() == 0,
              "Can't mix and match use of debug cloud indexing");

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
        point.rgba = 0xffffffff;

        point.z = depth[off];

        if (!std::isnormal(point.z))
            continue;

        point.x = (x - cx) * point.z * inv_fx;

        /* NB: 2D depth coords have y=0 at the top, and we want +Y to extend
         * upwards...
         */
        point.y = -((y - cy) * point.z * inv_fy);

        debug_cloud.push_back(point);
    }
}

static void
add_debug_cloud_xyz_from_codebook(struct gm_context *ctx,
                                  struct gm_tracking_impl *tracking,
                                  std::vector<std::vector<struct seg_codeword>> &seg_codebook,
                                  std::vector<struct seg_codeword *> &seg_codebook_bg,
                                  struct gm_intrinsics *intrinsics)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    gm_assert(ctx->log, debug_cloud_indices.size() == 0,
              "Can't mix and match use of debug cloud indexing");

    int width = intrinsics->width;
    int height = intrinsics->height;

    const float fx = intrinsics->fx;
    const float fy = intrinsics->fy;
    const float inv_fx = 1.0f / fx;
    const float inv_fy = 1.0f / fy;
    const float cx = intrinsics->cx;
    const float cy = intrinsics->cy;

    int debug_layer = ctx->debug_codebook_layer;

    foreach_xy_off(width, height) {
        std::vector<struct seg_codeword> &codewords = seg_codebook[off];
        struct gm_point_rgba point;

        point.rgba = 0xffffffff;

        for (int i = 0; i < (int)codewords.size(); i++) {
            struct seg_codeword &codeword = codewords[i];

            if (debug_layer < 0) {
                if (i != (codewords.size() + debug_layer))
                    continue;
            } else if (debug_layer > 0) {
                if (i != (debug_layer - 1))
                    continue;
            }

            point.z = codeword.mean;

            point.x = (x - cx) * point.z * inv_fx;

            /* NB: 2D depth coords have y=0 at the top, and we want +Y to
             * extend upwards...
             */
            point.y = -((y - cy) * point.z * inv_fy);

            if (&codeword == seg_codebook_bg[off])
                point.rgba = 0xff0000ff;

            debug_cloud.push_back(point);
        }
    }
}

static void
add_debug_cloud_person_masks_except(struct gm_tracking_impl *tracking,
                                    struct pipeline_scratch_state *state,
                                    int except_person)

{
    struct gm_context *ctx = tracking->ctx;

    int except_label = (except_person < 0 ? -1 :
                        state->person_clusters[except_person].label);

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;

    for (auto &cluster : state->person_clusters) {
        if (cluster.label == except_label)
            continue;

        add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                      tracking->downsampled_cloud,
                                                      cluster_indices[cluster.label].indices);
    }
}

static void
add_debug_cloud_xyz_of_codebook_space(struct gm_context *ctx,
                                      struct gm_tracking_impl *tracking,
                                      pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud,
                                      glm::mat4 to_start,
                                      glm::mat4 start_to_codebook,
                                      struct gm_intrinsics *intrinsics)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    gm_assert(ctx->log, debug_cloud.size() == debug_cloud_indices.size(),
              "Can't mix and match use of debug cloud indexing");

    for (unsigned i = 0; i < pcl_cloud->size(); i++) {
        pcl::PointXYZL pcl_point = pcl_cloud->points[i];
        struct gm_point_rgba point;

        int off = project_point_into_codebook(&pcl_point,
                                              to_start,
                                              start_to_codebook,
                                              intrinsics);
        // Falls outside of codebook...
        if (off < 0)
            continue;

        point.x = pcl_point.x;
        point.y = pcl_point.y;
        point.z = pcl_point.z;
        point.rgba = 0xffffffff;

        debug_cloud.push_back(point);
        debug_cloud_indices.push_back(i);
    }
}

/* Used to colour the debug cloud for most pipeline stages... */
static void
colour_debug_cloud(struct gm_context *ctx,
                   struct pipeline_scratch_state *state,
                   struct gm_tracking_impl *tracking,
                   pcl::PointCloud<pcl::PointXYZL>::Ptr indexed_pcl_cloud)
{
    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &indices = tracking->debug_cloud_indices;

    gm_assert(ctx->log,
              (indices.size() == 0 || debug_cloud.size() == indices.size()),
              "Can't mix and match use of debug cloud indexing");

    switch ((enum debug_cloud_mode)state->debug_cloud_mode)
    {
    case DEBUG_CLOUD_MODE_VIDEO: {
        const float vid_fx = tracking->video_camera_intrinsics.fx;
        const float vid_fy = tracking->video_camera_intrinsics.fy;
        const float vid_cx = tracking->video_camera_intrinsics.cx;
        const float vid_cy = tracking->video_camera_intrinsics.cy;

        int vid_width = 0;
        int vid_height = 0;
        uint8_t *vid_rgb = NULL;
        tracking_create_rgb_video(&tracking->base, &vid_width, &vid_height, &vid_rgb);
        if (vid_rgb) {
            if (indexed_pcl_cloud && indices.size()) {
                for (int i = 0; i < indices.size(); i++) {
                    int idx = indices[i];

                    gm_assert(ctx->log, idx < indexed_pcl_cloud->size(),
                              "Out-of-bounds debug point cloud index (%d, n_points = %d)",
                              idx, (int)indexed_pcl_cloud->size());

                    float x = indexed_pcl_cloud->points[idx].x;
                    float y = indexed_pcl_cloud->points[idx].y;
                    float z = indexed_pcl_cloud->points[idx].z;

                    if (!std::isnormal(z))
                        continue;

                    // Reproject the depth coordinates into video space
                    // TODO: Support extrinsics
                    int vx = clampf(x * vid_fx / z + vid_cx, 0.0f, (float)vid_width - 1);

                    // NB: 2D tex coords have y=0 at the top, extending down
                    // while we have y+ extending upwards, so we need to flip
                    // when mapping back to 2D coords...
                    int vy = clampf(y * -vid_fy / z + vid_cy, 0.0f,
                                    (float)vid_height - 1);
                    int v_off = vy * vid_width * 3 + vx * 3;

                    debug_cloud[i].rgba = (((uint32_t)vid_rgb[v_off])<<24 |
                                           ((uint32_t)vid_rgb[v_off+1])<<16 |
                                           ((uint32_t)vid_rgb[v_off+2])<<8 |
                                           0xff);
                }
            } else {
                for (unsigned off = 0; off < debug_cloud.size(); off++) {
                    float x = debug_cloud[off].x;
                    float y = debug_cloud[off].y;
                    float z = debug_cloud[off].z;

                    if (!std::isnormal(z))
                        continue;

                    // Reproject the depth coordinates into video space
                    // TODO: Support extrinsics
                    int vx = clampf(x * vid_fx / z + vid_cx, 0.0f, (float)vid_width - 1);

                    // NB: 2D tex coords have y=0 at the top, extending down
                    // while we have y+ extending upwards, so we need to flip
                    // when mapping back to 2D coords...
                    int vy = clampf(y * -vid_fy / z + vid_cy, 0.0f,
                                    (float)vid_height - 1);
                    int v_off = vy * vid_width * 3 + vx * 3;

                    debug_cloud[off].rgba = (((uint32_t)vid_rgb[v_off])<<24 |
                                             ((uint32_t)vid_rgb[v_off+1])<<16 |
                                             ((uint32_t)vid_rgb[v_off+2])<<8 |
                                             0xff);
                }
            }

            free(vid_rgb);
        }
        break;
    }
    case DEBUG_CLOUD_MODE_DEPTH:
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
        break;
    case DEBUG_CLOUD_MODE_CODEBOOK_LABELS:
        if (state->codebook_classified && indices.size()) {
            for (unsigned i = 0; i < indices.size(); i++) {
                enum codebook_class label =
                    (enum codebook_class)indexed_pcl_cloud->points[indices[i]].label;
                uint8_t rgb[3];
                depth_classification_to_rgb(label, rgb);
                debug_cloud[i].rgba = (((uint32_t)rgb[0])<<24 |
                                       ((uint32_t)rgb[1])<<16 |
                                       ((uint32_t)rgb[2])<<8 |
                                       0xff);
            }
        }
        break;
    case DEBUG_CLOUD_MODE_LABELS:
        if (state->done_label_inference)
        {
            int n_labels = ctx->n_labels;

            if (debug_cloud.size() * n_labels == tracking->label_probs.size()) {
                for (int i = 0; i < debug_cloud.size(); i++) {
                    float *label_probs = &tracking->label_probs[i * n_labels];
                    uint8_t rgb[3];
                    label_probs_to_rgb(ctx, label_probs, n_labels, rgb);
                    debug_cloud[i].rgba = (((uint32_t)rgb[0])<<24 |
                                           ((uint32_t)rgb[1])<<16 |
                                           ((uint32_t)rgb[2])<<8 |
                                           0xff);
                }
            } else {
                gm_warn(ctx->log, "Can't color debug cloud with labels due to inconsistent point cloud and label_probs[] size.");
            }
        }
        break;
    case DEBUG_CLOUD_MODE_EDGES:
        if (state->done_edge_detect &&
            state->debug_pipeline_stage == TRACKING_STAGE_EDGE_DETECT)
        {
            std::vector<bool> &edge_mask = ctx->edge_detect_scratch;
            for (int i = 0; i < debug_cloud.size(); i++) {
                if (edge_mask[i]) {
                    debug_cloud[i].rgba = 0xffffffff;
                } else {
                    debug_cloud[i].rgba = 0x202020ff;
                }
            }
        }
        break;
    case DEBUG_CLOUD_MODE_NONE:
    case N_DEBUG_CLOUD_MODES:
        gm_assert(ctx->log, 0, "Shouldn't be reached: spurious cloud mode value");
        break;
    }
}

static void
stage_start_cb(struct gm_tracking_impl *tracking,
                 struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    struct gm_frame *frame = tracking->frame;

    copy_and_rotate_depth_buffer(ctx,
                                 tracking,
                                 &frame->depth_intrinsics,
                                 frame->depth_format,
                                 frame->depth);
}

static void
stage_start_debug_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_dense_depth_buf(ctx, tracking,
                                             tracking->depth,
                                             &tracking->depth_camera_intrinsics);
    tracking->debug_cloud_intrinsics = tracking->depth_camera_intrinsics;
    colour_debug_cloud(ctx, state, tracking, NULL);

    if (tracking->frame->gravity_valid) {
        float start[3] = { 0, 0, 1 };
        float *gravity = tracking->frame->gravity;
        float end[3] = {
            start[0] + gravity[0],
            start[1] + gravity[1],
            start[2] + gravity[2]
        };

        float pos[3] = { 0, 0, 1 };
        uint32_t colors[3] = { 0xff0000ff, 0x00ff00ff, 0x0000ffff };
        tracking_draw_transformed_axis(tracking,
                                       pos,
                                       colors,
                                       glm::mat4(1.0));

        tracking_draw_line(tracking,
                           start[0], start[1], start[2],
                           end[0], end[1], end[2],
                           0x008080ff);
    }
}

static void
stage_near_far_cull_and_infill_cb(struct gm_tracking_impl *tracking,
                                  struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    if (!tracking->depth_cloud) {
        tracking->depth_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(
            new pcl::PointCloud<pcl::PointXYZL>);
    }

    pcl_xyzl_cloud_from_buf_with_near_far_cull_and_infill(ctx,
                                                          tracking,
                                                          tracking->depth_cloud,
                                                          tracking->depth,
                                                          &tracking->
                                                          depth_camera_intrinsics);
}

static void
stage_near_far_cull_and_infill_debug_cb(struct gm_tracking_impl *tracking,
                                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->depth_cloud);
    tracking->debug_cloud_intrinsics = tracking->depth_camera_intrinsics;
    colour_debug_cloud(ctx, state, tracking, tracking->depth_cloud);
}

static void
stage_downsample_cb(struct gm_tracking_impl *tracking,
                    struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;

    // Person detection can happen in a sparser cloud made from a downscaled
    // version of the depth buffer. This is significantly cheaper than using a
    // voxel grid, which would produce better results but take a lot longer
    // doing so and give us less useful data structures.
    int seg_res = state->seg_res;
    if (seg_res == 1) {
        tracking->downsampled_cloud = tracking->depth_cloud;
        tracking->downsampled_intrinsics = tracking->depth_camera_intrinsics;
    } else {
        if (!tracking->downsampled_cloud ||
            tracking->downsampled_cloud == tracking->depth_cloud) {
            tracking->downsampled_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(
                new pcl::PointCloud<pcl::PointXYZL>);
        }

        tracking->downsampled_cloud->width = tracking->depth_cloud->width / seg_res;
        tracking->downsampled_cloud->height = tracking->depth_cloud->height / seg_res;
        tracking->downsampled_cloud->points.resize(tracking->downsampled_cloud->width *
                                                   tracking->downsampled_cloud->height);
        tracking->downsampled_cloud->is_dense = false;

        int n_lores_points = 0;
        foreach_xy_off(tracking->downsampled_cloud->width,
                       tracking->downsampled_cloud->height) {
            int hoff = (y * seg_res) * tracking->depth_cloud->width +
                (x * seg_res);
            tracking->downsampled_cloud->points[off].x =
                tracking->depth_cloud->points[hoff].x;
            tracking->downsampled_cloud->points[off].y =
                tracking->depth_cloud->points[hoff].y;
            tracking->downsampled_cloud->points[off].z =
                tracking->depth_cloud->points[hoff].z;
            tracking->downsampled_cloud->points[off].label = -1;
            if (!std::isnan(tracking->downsampled_cloud->points[off].z)) {
                ++n_lores_points;
            }
        }

        tracking->downsampled_intrinsics = tracking->depth_camera_intrinsics;
        tracking->downsampled_intrinsics.width /= seg_res;
        tracking->downsampled_intrinsics.height /= seg_res;
        tracking->downsampled_intrinsics.cx /= seg_res;
        tracking->downsampled_intrinsics.cy /= seg_res;
        tracking->downsampled_intrinsics.fx /= seg_res;
        tracking->downsampled_intrinsics.fy /= seg_res;
    }
}

static void
stage_downsample_debug_cb(struct gm_tracking_impl *tracking,
                          struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

    float pos[3] = { 0, 0, 1 };
    uint32_t colors[3] = { 0xff0000ff, 0x00ff00ff, 0x0000ffff };
    tracking_draw_transformed_axis(tracking,
                                   pos,
                                   colors,
                                   glm::mat4(1.0));

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
}

/* 'edge' in this context specifically means finding gradients that *very*
 * closely match the viewing angle at that point.
 *
 * We are primarily concerned with the significant artefacts that are seen in
 * iPhone depth buffers where it looks like geometry is smeared/interpolated at
 * the edges of objects which really interferes with clustering neighbouring
 * points based on distance thresholding (everything is closely connected).
 */
static void
stage_edge_detect_cb(struct gm_tracking_impl *tracking,
                     struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    int width = tracking->downsampled_cloud->width;
    int height = tracking->downsampled_cloud->height;

    pcl::PointCloud<pcl::PointXYZL>::VectorType &points =
        tracking->downsampled_cloud->points;

    std::vector<bool> &edge_mask = ctx->edge_detect_scratch;
    edge_mask.resize(width * height);

    bool x_edges = false;
    bool y_edges = false;

    switch ((enum edge_detect_mode)ctx->edge_detect_mode)
    {
    case EDGE_DETECT_MODE_X_ONLY:
        x_edges = true;
        break;
    case EDGE_DETECT_MODE_Y_ONLY:
        y_edges = true;
        break;
    case EDGE_DETECT_MODE_XY:
        x_edges = true;
        y_edges = true;
        break;
    case EDGE_DETECT_MODE_NONE:
        break;
    }

    int edge_break = -1;

    if (ctx->edge_break_x >= 0 &&
        ctx->edge_break_x < width &&
        ctx->edge_break_y >=0 &&
        ctx->edge_break_y < height)
    {
        edge_break = ctx->edge_break_y * width + ctx->edge_break_x;
    }

    float edge_threshold = ctx->edge_threshold;

    // XXX: we don't use memset because vector<bool> is rather special and
    // implemented as an array of bits whose size we don't really know.
    std::fill(edge_mask.begin(), edge_mask.end(), 0);

    if (x_edges) {
        foreach_xy_off(width, height) {
            pcl::PointXYZL &point = points[off];
            if (std::isnan(point.z)) {
                edge_mask[off] = 1;
                continue;
            }
            if (x == 0 || x == width - 1) {
                continue;
            }

            /* XXX: look at thresholding based on the squared distances instead
             * so we can avoid normalizing 3 vectors for every pixel
             *
             * XXX: also look at working with a scanline since point becomes
             * point_l and point_r becomes point for the next iteration so we
             * could avoid one of the normalizations each horizontal step.
             *
             * XXX: really we only need to use glm::vec2() for this
             */

            glm::vec3 eye = glm::normalize(glm::vec3(point.x, 0, point.z));

            pcl::PointXYZL &point_l = points[off-1];

            glm::vec3 grad_l = glm::normalize(
                glm::vec3(point_l.x, 0, point_l.z) -
                glm::vec3(point.x, 0, point.z));

            float compare = glm::dot(grad_l, eye);
            if (compare > edge_threshold) {
                edge_mask[off] = 1;
            } else {
                pcl::PointXYZL &point_r = points[off+1];

                glm::vec3 grad_r = glm::normalize(
                    glm::vec3(point_r.x, 0, point_r.z) -
                    glm::vec3(point.x, 0, point.z));

                float compare = glm::dot(grad_r, eye);
                if (compare > edge_threshold) {
                    edge_mask[off] = 1;
                }
            }
        }
    }

    if (y_edges) {
        foreach_xy_off(width, height) {
            pcl::PointXYZL &point = points[off];
            if (std::isnan(point.z)) {
                edge_mask[off] = 1;
                continue;
            }
            if (y == 0 || y == height - 1) {
                continue;
            }

            pcl::PointXYZL &point_u = points[off-width];

            /* XXX: look at thresholding based on the squared distances instead
             * so we can avoid normalizing 3 vectors for every pixel
             *
             * XXX: also look at working with a scanline since point becomes
             * point_l and point_r becomes point for the next iteration so we
             * could avoid one of the normalizations each horizontal step.
             *
             * XXX: really we only need to use glm::vec2() for this
             */

            glm::vec3 eye = glm::normalize(glm::vec3(0, point.y, point.z));

            glm::vec3 grad_u = glm::normalize(
                glm::vec3(0, point_u.y, point_u.z) -
                glm::vec3(0, point.y, point.z));

            float compare = glm::dot(grad_u, eye);
            if (fabs(compare) > edge_threshold) {
                edge_mask[off] = 1;
            } else {
                pcl::PointXYZL &point_d = points[off+width];

                glm::vec3 grad_d = glm::normalize(
                    glm::vec3(0, point_d.y, point_d.z) -
                    glm::vec3(0, point.y, point.z));
                float compare = glm::dot(grad_d, eye);
                if (fabs(compare) > edge_threshold) {
                    edge_mask[off] = 1;
                }
            }
        }
    }

    /* We might be running this stage even if !ctx->delete_edges, just
     * for the debug visualization of what would be deleted...
     */
    if (ctx->delete_edges) {
        float nan = std::numeric_limits<float>::quiet_NaN();

        foreach_xy_off(width, height) {
            if (edge_mask[off]) {
                pcl::PointXYZL &point = points[off];
                point.x = point.y = point.z = nan;
                point.label = -1;
            }
        }
    }

    state->done_edge_detect = true;
}

static void
stage_edge_detect_debug_cb(struct gm_tracking_impl *tracking,
                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    int width = tracking->downsampled_cloud->width;
    int height = tracking->downsampled_cloud->height;

    pcl::PointCloud<pcl::PointXYZL>::VectorType &points =
        tracking->downsampled_cloud->points;

    std::vector<bool> &edge_mask = ctx->edge_detect_scratch;

    int edge_break = -1;

    if (ctx->edge_break_x >= 0 &&
        ctx->edge_break_x < width &&
        ctx->edge_break_y >=0 &&
        ctx->edge_break_y < height)
    {
        edge_break = ctx->edge_break_y * width + ctx->edge_break_x;
    }

    if (edge_break >= 0 && edge_break < edge_mask.size()) {
        int off = edge_break;

        pcl::PointXYZL &point = points[off];
        if (std::isnan(point.z)) {
            return;
        }

        float pos[3] = { point.x, point.y, point.z };
        tracking_draw_transformed_crosshair(
            tracking,
            pos,
            0.01, // 1cm
            0x00ff00ff,
            glm::mat4(1.0));

        int x = off % width;
        if (x == 0 || x == width - 1) {
            return;
        }
        pcl::PointXYZL &point_l = points[off-1];
        pcl::PointXYZL &point_r = points[off+1];

        glm::vec3 eye = glm::normalize(glm::vec3(point.x, 0, point.z));

        glm::vec3 grad_l = glm::normalize(
            glm::vec3(point_l.x, 0, point_l.z) -
            glm::vec3(point.x, 0, point.z));
        glm::vec3 grad_r = glm::normalize(
            glm::vec3(point_r.x, 0, point_r.z) -
            glm::vec3(point.x, 0, point.z));

        tracking_draw_line(tracking,
                           0, 0, 0,
                           eye.x, eye.y, eye.z,
                           0xffffffff);
        tracking_draw_line(tracking,
                           0, 0, 0,
                           grad_l.x, grad_l.y, grad_l.z,
                           0xff0000ff);
        tracking_draw_line(tracking,
                           0, 0, 0,
                           grad_r.x, grad_r.y, grad_r.z,
                           0x00ff00ff);
    }
}

static void
stage_ground_project_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;
    glm::mat4 to_ground = state->to_ground;

    // Transform the cloud into ground-aligned space if we have a valid pose
    if (!tracking->ground_cloud) {
        tracking->ground_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(
            new pcl::PointCloud<pcl::PointXYZL>);
    }
    if (state->to_ground_valid) {
        unsigned downsampled_cloud_size = tracking->downsampled_cloud->points.size();

        int width = tracking->downsampled_cloud->width;
        int height = tracking->downsampled_cloud->height;

        tracking->ground_cloud->width = width;
        tracking->ground_cloud->height = height;
        tracking->ground_cloud->points.resize(downsampled_cloud_size);
        tracking->ground_cloud->is_dense = false;

        float nan = std::numeric_limits<float>::quiet_NaN();
        pcl::PointXYZL invalid_pt;
        invalid_pt.x = invalid_pt.y = invalid_pt.z = nan;
        invalid_pt.label = -1;

        foreach_xy_off(width, height) {
            pcl::PointXYZL &point = tracking->downsampled_cloud->points[off];
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

    if (state->to_ground_valid) {
        add_debug_cloud_xyz_from_pcl_xyzl_transformed(ctx, tracking,
                                                      tracking->downsampled_cloud,
                                                      state->to_ground);
        colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
        float pos[3] = { 0, 0, 1 };
        uint32_t colors[3] = { 0xff0000ff, 0x00ff00ff, 0x0000ffff };
        tracking_draw_transformed_axis(tracking,
                                       pos,
                                       colors,
                                       state->to_ground);

        uint32_t light_colors[3] = { 0x800000ff, 0x008000ff, 0x000080ff };
        tracking_draw_transformed_axis(tracking,
                                       pos,
                                       light_colors,
                                       glm::mat4(1.0));

        if (tracking->frame->gravity_valid) {
            float start[3] = { 0, 0, 1 };
            float *gravity = tracking->frame->gravity;
            float end[3] = {
                start[0] + gravity[0],
                start[1] + gravity[1],
                start[2] + gravity[2]
            };

            tracking_draw_transformed_line(tracking,
                                           start,
                                           end,
                                           0x00ffffff,
                                           state->to_ground);
            tracking_draw_line(tracking,
                               start[0], start[1], start[2],
                               end[0], end[1], end[2],
                               0x008080ff);
        }
    } else {
        add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);
        colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
    }

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_codebook_retire_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    if (state->codebook_frozen)
        return;

    struct gm_context *ctx = tracking->ctx;

    std::vector<std::vector<struct seg_codeword>> &seg_codebook =
        *state->seg_codebook;
    unsigned codebook_size = tracking->downsampled_cloud->points.size();

    uint64_t frame_timestamp = tracking->frame->timestamp;

    uint64_t foreground_scrub_timeout =
        (double)ctx->codebook_foreground_scrub_timeout * 1e9;

    uint64_t clear_timeout =
        (double)ctx->codebook_clear_timeout * 1e9;

    uint64_t since_tracked_duration =
        frame_timestamp - ctx->last_tracking_success_timestamp;

    if (!ctx->latest_tracking || !ctx->latest_tracking->success)
    {
        if (frame_timestamp - ctx->codebook_last_clear_timestamp > clear_timeout &&
            since_tracked_duration > clear_timeout)
        {
            state->seg_codebook->clear();
            state->seg_codebook->resize(codebook_size);
            if (!state->paused)
                ctx->codebook_last_clear_timestamp = frame_timestamp;
        } else if (frame_timestamp - ctx->codebook_last_foreground_scrub_timestamp > foreground_scrub_timeout &&
                   since_tracked_duration > foreground_scrub_timeout)
        {
            // Considering that the codebook may be polluted at this point by a
            // human failing to track we try to remove all but the
            // furthest-away codewords. This is in the hope that if there was
            // an untracked human in the codebook that at some point we saw
            // background behind them.

            std::vector<std::vector<struct seg_codeword>> &seg_codebook =
                *state->seg_codebook;

            float keep_back_most_threshold = ctx->codebook_keep_back_most_threshold;
            unsigned codebook_size = seg_codebook.size();

            for (unsigned off = 0; off < codebook_size; off++) {
                std::vector<struct seg_codeword> &codewords = seg_codebook[off];

                if (!codewords.size())
                    continue;

                // NB: the codebook is sorted from nearest to farthest
                int n_codewords = (int)codewords.size();
                float back_most = codewords[n_codewords-1].mean;

                for (int i = n_codewords - 1; i >= 0; i--) {
                    if (fabsf(codewords[i].mean - back_most) > keep_back_most_threshold)
                    {
                        int j = 0;
                        for (i++; i < n_codewords; i++) {
                            codewords[j++] = codewords[i];
                        }
                        codewords.resize(j);
                        break;
                    }
                }
            }

            if (!state->paused)
                ctx->codebook_last_foreground_scrub_timestamp = frame_timestamp;
        }
    }

    uint64_t codeword_timeout_ns = ctx->codeword_timeout * 1e9;

    for (unsigned off = 0; off < codebook_size; off++) {
        std::vector<struct seg_codeword> &codewords = seg_codebook[off];

        for (unsigned i = 0; i < codewords.size();) {
            struct seg_codeword &codeword = codewords[i];

            if ((frame_timestamp - codeword.last_update_timestamp) >= codeword_timeout_ns)
                codewords.erase(codewords.begin() + i);
            else
                i++;
        }
    }
}

static void
stage_codebook_retire_debug_cb(struct gm_tracking_impl *tracking,
                               struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;

    if (ctx->codebook_debug_view == CODEBOOK_DEBUG_VIEW_POINT_CLOUD) {
        add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

        colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
    } else {
        add_debug_cloud_xyz_from_codebook(ctx,
                                          tracking,
                                          *state->seg_codebook,
                                          ctx->seg_codebook_bg,
                                          &tracking->debug_cloud_intrinsics);
    }
}

static void
stage_codebook_resolve_background_cb(struct gm_tracking_impl *tracking,
                                     struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    std::vector<std::vector<struct seg_codeword>> &seg_codebook =
        *state->seg_codebook;
    std::vector<struct seg_codeword *> &seg_codebook_bg = ctx->seg_codebook_bg;
    unsigned codebook_size = tracking->downsampled_cloud->points.size();

    seg_codebook_bg.resize(codebook_size);

    // Ranked by codeword.n (the number of times points have matched it) pick
    // the codeword with the highest number of matches as our canonical
    // background codeword.
    //
    // Note: if we find multiple codewords with the same ->n count then as a
    // tie breaker we pick the codeword with the farthest mean depth as the
    // canonical background.
    //
    for (unsigned off = 0; off < codebook_size; off++) {
        std::vector<struct seg_codeword> &codewords = seg_codebook[off];

        // Note: we have to be careful to not allow edits of the
        // codebook while maintaining seg_codebook_bg to avoid
        // invalidating these codeword pointers!
        //
        seg_codebook_bg[off] = NULL;
        for (unsigned i = 0; i < codewords.size(); i++) {
            struct seg_codeword &codeword = codewords[i];

            if (!ctx->seg_codebook_bg[off] ||
                codeword.n > seg_codebook_bg[off]->n ||
                (codeword.n == seg_codebook_bg[off]->n &&
                 codeword.mean > seg_codebook_bg[off]->mean))
            {
                seg_codebook_bg[off] = &codeword;
            }
        }
    }
}

static void
stage_codebook_resolve_background_debug_cb(struct gm_tracking_impl *tracking,
                                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;

    if (ctx->codebook_debug_view == CODEBOOK_DEBUG_VIEW_POINT_CLOUD) {
        add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

        colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
    } else {
        add_debug_cloud_xyz_from_codebook(ctx,
                                          tracking,
                                          *state->seg_codebook,
                                          ctx->seg_codebook_bg,
                                          &tracking->debug_cloud_intrinsics);
    }
}

static void
stage_codebook_project_debug_cb(struct gm_tracking_impl *tracking,
                                struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    glm::mat4 to_start = state->to_start;
    glm::mat4 start_to_codebook = state->start_to_codebook;

    add_debug_cloud_xyz_of_codebook_space(
        ctx, tracking, tracking->downsampled_cloud, to_start,
        start_to_codebook, &tracking->downsampled_intrinsics);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_codebook_classify_cb(struct gm_tracking_impl *tracking,
                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    std::vector<std::vector<struct seg_codeword>> &seg_codebook =
        *state->seg_codebook;
    std::vector<struct seg_codeword *> &seg_codebook_bg = ctx->seg_codebook_bg;
    glm::mat4 to_start = state->to_start;
    glm::mat4 start_to_codebook = state->start_to_codebook;
    unsigned downsampled_cloud_size = tracking->downsampled_cloud->points.size();
    uint64_t frame_timestamp = 0;
    uint64_t frame_counter = 0;

    /* If the codebook is frozen then we want any classification that is timing
     * sensitive also be based on a frozen timestamp...
     */
    if (state->codebook_frozen) {
        frame_timestamp = ctx->last_codebook_update_time;
        frame_counter = ctx->last_codebook_update_frame_counter;
    } else {
        frame_timestamp = tracking->frame->timestamp;
        frame_counter = state->frame_counter;
    }

    pcl::PointCloud<pcl::PointXYZL>::VectorType &downsampled_points =
        tracking->downsampled_cloud->points;

    struct gm_intrinsics codebook_intrinsics = tracking->downsampled_intrinsics;

    const float codebook_bg_threshold = ctx->codebook_bg_threshold;
    const float codebook_flat_threshold = ctx->codebook_flat_threshold;
    const int codeword_flicker_max_run_len = ctx->codeword_flicker_max_run_len;
    const int codeword_flicker_max_quiet_frames = (float)ctx->codeword_flicker_max_quiet_frames;
    const int codeword_obj_min_n = ctx->codeword_obj_min_n;
    const float codeword_obj_max_frame_to_n_ratio = ctx->codeword_obj_max_frame_to_n_ratio;

    for (unsigned depth_off = 0; depth_off < downsampled_cloud_size; depth_off++)
    {
        pcl::PointXYZL point = downsampled_points[depth_off];

        if (std::isnan(point.z)) {
            // We'll never cluster a nan value, so we can immediately
            // classify it as background.
            downsampled_points[depth_off].label = CODEBOOK_CLASS_BACKGROUND;
            continue;
        }

        int off = project_point_into_codebook(&point,
                                              to_start,
                                              start_to_codebook,
                                              &codebook_intrinsics);
        // Falls outside of codebook so we can't classify...
        if (off < 0)
            continue;

        // At this point z has been projected into the coordinate space
        // of the codebook
        float depth = point.z;

        // Look to see if this pixel falls into an existing codeword
        struct seg_codeword *codeword = NULL;
        float best_codeword_distance = FLT_MAX;
        struct seg_codeword *bg_codeword = seg_codebook_bg[off];

        std::vector<struct seg_codeword> &codewords = seg_codebook[off];
        for (auto &candidate : codewords) {
            /* The codewords are sorted from closest to farthest */
            float dist = fabsf(depth - candidate.mean);
            if (dist < best_codeword_distance) {
                codeword = &candidate;
                best_codeword_distance = dist;
            } else {
                // Any other codewords will be even farther away
                break;
            }
        }

        if (best_codeword_distance > codebook_bg_threshold)
            codeword = NULL;

        gm_assert(ctx->log,
                  bg_codeword || (!bg_codeword && !codeword),
                  "If no default background codeword, we shouldn't match any codeword based on mean distance");

        if (!codeword) {
            downsampled_points[depth_off].label = CODEBOOK_CLASS_FOREGROUND;
            continue;
        }

        if (codeword == bg_codeword) {
            downsampled_points[depth_off].label = CODEBOOK_CLASS_BACKGROUND;
            continue;
        }

        float dist_from_background = fabsf(codeword->mean - bg_codeword->mean);

        if (dist_from_background < codebook_bg_threshold) {
            downsampled_points[depth_off].label = CODEBOOK_CLASS_BACKGROUND;
            continue;
        }

        bool flat = false;
        bool flickering = false;

        /* Note: from the _BACKGROUND check above we already know that
         * dist_from_background is > codebook_bg_threshold
         */
        if (dist_from_background <= codebook_flat_threshold) {
            flat = true;
        }

        // 'Flickering' is defined based on two conditions:
        //
        //  1) the codeword doesn't (on average) see runs of more than
        //     ->codeword_flicker_max_run_len consecutive updates
        //  2) The first condition should be met (on average) at least every
        //     N frames (where N = 'codeword_flicker_max_quiet_frames')
        //
        bool requirement_one = (codeword_flicker_max_run_len *
                                codeword->n_consecutive_update_runs) > codeword->n;

        int n_frames_since_create = frame_counter - codeword->create_frame_counter;
        bool requirement_two =
            ((n_frames_since_create / codeword_flicker_max_quiet_frames) <=
             codeword->n_consecutive_update_runs);
        if (requirement_one && requirement_two)
            flickering = true;

        if (flat || flickering) {
            downsampled_points[depth_off].label =
                (flat && flickering) ? CODEBOOK_CLASS_FLAT_AND_FLICKERING :
                (flat ? CODEBOOK_CLASS_FLAT : CODEBOOK_CLASS_FLICKERING);
            continue;
        }

        int n_update_frames = (codeword->last_update_frame_count -
                               codeword->create_frame_counter);
        if (codeword->n > codeword_obj_min_n &&
            n_update_frames / (float)codeword->n >= codeword_obj_max_frame_to_n_ratio)
        {
            downsampled_points[depth_off].label = CODEBOOK_CLASS_FOREGROUND_OBJ_TO_IGNORE;
        } else {
            downsampled_points[depth_off].label = CODEBOOK_CLASS_FOREGROUND;
        }
    }

    state->codebook_classified = true;
}

static void
stage_codebook_classify_debug_cb(struct gm_tracking_impl *tracking,
                                 struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;

    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
}

// When scanning along a single row we may define new labels due to comparison
// failures which we later find need to be effectively merged with the next
// row. This merging is done by maintaining an array of label ids ('runs')
// where normally runs[label_id] == label_id but when we need to track the
// merging of label IDs then runs[label_id] can point back to an earlier
// label_id index. There may be multiple jumps like this due to multiple such
// merges so this function follows the indirections to find the 'root' label_id
// that will be the final effective label_id.
static unsigned
find_label_root(std::vector<unsigned>& runs, unsigned index)
{
    unsigned idx = index;
    while (runs[idx] != idx)
        idx = runs[idx];

    // In case there were multiple indirections then updating the index
    // will reduce all further lookups to one indirection at most
    //runs[index] = idx; (Didn't improve performance)

    return idx;
}

static bool
compare_codebook_classified_points(
    pcl::PointCloud<pcl::PointXYZL>::Ptr input_,
    int idx1, int idx2,
    float depth_threshold)
{
    if ((input_->points[idx1].label == CODEBOOK_CLASS_FLICKERING ||
         input_->points[idx1].label == CODEBOOK_CLASS_FOREGROUND) &&
        (input_->points[idx2].label == CODEBOOK_CLASS_FLICKERING ||
         input_->points[idx2].label == CODEBOOK_CLASS_FOREGROUND))
    {
        return fabsf(input_->points[idx1].z - input_->points[idx2].z) <
            depth_threshold;
    }

    return false;
}

static void
cluster_codebook_classified_points(
    pcl::PointCloud<pcl::PointXYZL>::Ptr input_,
    pcl::PointCloud<pcl::Label>& labels,
    std::vector<pcl::PointIndices>& label_indices,
    float depth_threshold)
{
    std::vector<unsigned> run_ids;

    unsigned invalid_label = std::numeric_limits<unsigned>::max();
    pcl::Label invalid_pt;
    invalid_pt.label = std::numeric_limits<unsigned>::max();
    labels.points.clear();
    labels.points.resize(input_->points.size(), invalid_pt);
    labels.width = input_->width;
    labels.height = input_->height;

    //First pixel
    if (std::isfinite(input_->points[0].x))
    {
        labels[0].label = run_ids.size();
        run_ids.push_back(labels[0].label);
    }

    // First row
    for (int colIdx = 1; colIdx < static_cast<int>(input_->width); ++colIdx)
    {
        if (!std::isfinite(input_->points[colIdx].x))
            continue;

        if (compare_codebook_classified_points(input_, colIdx, colIdx - 1, depth_threshold))
        {
            unsigned label = labels[colIdx - 1].label;
            labels[colIdx].label = label;
        }
        else
        {
            labels[colIdx].label = run_ids.size();
            run_ids.push_back(labels[colIdx].label);
        }
    }

    // Everything else
    unsigned int current_row = input_->width;
    unsigned int previous_row = 0;
    for (size_t rowIdx = 1;
         rowIdx < input_->height;
         ++rowIdx, previous_row = current_row, current_row += input_->width)
    {
        // First pixel
        if (std::isfinite(input_->points[current_row].x))
        {
            if (compare_codebook_classified_points(input_,
                                                   current_row,
                                                   previous_row,
                                                   depth_threshold))
            {
                unsigned label = labels[previous_row].label;
                labels[current_row].label = label;
            }
            else
            {
                labels[current_row].label = run_ids.size();
                run_ids.push_back(labels[current_row].label);
            }
        }

        // Rest of row
        for (int colIdx = 1;
             colIdx < static_cast<int>(input_->width);
             ++colIdx)
        {
            if (std::isfinite(input_->points[current_row + colIdx].x))
            {
                if (compare_codebook_classified_points(input_,
                                                       current_row + colIdx,
                                                       current_row + colIdx - 1,
                                                       depth_threshold))
                {
                    unsigned label = labels[current_row + colIdx - 1].label;
                    labels[current_row + colIdx].label = label;
                }

                if (compare_codebook_classified_points(input_,
                                                       current_row + colIdx,
                                                       previous_row + colIdx,
                                                       depth_threshold))
                {
                    if (labels[current_row + colIdx].label == invalid_label)
                    {
                        unsigned label = labels[previous_row + colIdx].label;
                        labels[current_row + colIdx].label = label;
                    }
                    else if (labels[previous_row + colIdx].label != invalid_label)
                    {
                        unsigned root1 = find_label_root(run_ids, labels[current_row + colIdx].label);
                        unsigned root2 = find_label_root(run_ids, labels[previous_row + colIdx].label);

                        if (root1 < root2)
                            run_ids[root2] = root1;
                        else
                            run_ids[root1] = root2;
                    }
                }

                if (labels[current_row + colIdx].label == invalid_label)
                {
                    labels[current_row + colIdx].label = run_ids.size();
                    run_ids.push_back(labels[current_row + colIdx].label);
                }
            }
        }
    }

    std::vector<unsigned> map(run_ids.size());
    unsigned max_id = 0;
    for (unsigned runIdx = 0; runIdx < run_ids.size(); ++runIdx)
    {
        // if it is its own root -> new region
        if (run_ids[runIdx] == runIdx)
            map[runIdx] = max_id++;
        else // assign this sub-segment to the region (root) it belongs
            map[runIdx] = map[find_label_root(run_ids, runIdx)];
    }

    label_indices.resize(max_id + 1);
    for (unsigned idx = 0; idx < input_->points.size(); idx++)
    {
        if (labels[idx].label != invalid_label)
        {
            labels[idx].label = map[labels[idx].label];
            label_indices[labels[idx].label].indices.push_back(idx);
        }
    }
}

static void
stage_codebook_cluster_cb(struct gm_tracking_impl *tracking,
                          struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    if (!ctx->codebook_cluster_labels_scratch) {
        ctx->codebook_cluster_labels_scratch =
            pcl::PointCloud<pcl::Label>::Ptr(new pcl::PointCloud<pcl::Label>);
    }

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    cluster_indices.clear();

    cluster_codebook_classified_points(tracking->downsampled_cloud,
                                       *ctx->codebook_cluster_labels_scratch,
                                       cluster_indices,
                                       ctx->codebook_cluster_tolerance);

    int tiny_cluster_threshold = ctx->codebook_tiny_cluster_threshold;
    int large_cluster_threshold = ctx->codebook_large_cluster_threshold;

    int width = tracking->downsampled_cloud->width;
    //int height = tracking->downsampled_cloud->height;

    std::vector<candidate_cluster> large_clusters = {};

    if (ctx->codebook_cluster_infill)
    {
        pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud = tracking->downsampled_cloud;
        pcl::PointCloud<pcl::Label> &labels = *ctx->codebook_cluster_labels_scratch;
        std::vector<unsigned> to_merge = {};

        for (unsigned label = 0; label < cluster_indices.size(); label++)
        {
            auto &cluster = cluster_indices[label];

            if (cluster.indices.size() < large_cluster_threshold)
                continue;

            struct candidate_cluster large_label = { label };

            to_merge.clear();
            for (int i : cluster.indices) {
                pcl::PointXYZL &point = pcl_cloud->points[i];
                int x = i % width;
                int y = i / width;

                if (x == 0 || y == 0)
                    continue;

                if (x > large_label.max_x_2d)
                    large_label.max_x_2d = x;
                if (x < large_label.min_x_2d)
                    large_label.min_x_2d = x;
                if (y > large_label.max_y_2d)
                    large_label.max_y_2d = y;
                if (y < large_label.min_y_2d)
                    large_label.min_y_2d = y;

                if (point.x > large_label.max_x)
                    large_label.max_x = point.x;
                if (point.x < large_label.min_x)
                    large_label.min_x = point.x;
                if (point.y > large_label.max_y)
                    large_label.max_y = point.y;
                if (point.y < large_label.min_y)
                    large_label.min_y = point.y;
                if (point.z > large_label.max_z)
                    large_label.max_z = point.z;
                if (point.z < large_label.min_z)
                    large_label.min_z = point.z;

                int left_neightbour =  y * width + (x - 1);
                unsigned left_label = labels[left_neightbour].label;
                if (left_label != label &&
                    left_label != -1 &&
                    cluster_indices[left_label].indices.size() < tiny_cluster_threshold)
                {
                    to_merge.push_back(left_label);
                }

                int top_neightbour =  (y - 1) * width + x;
                unsigned top_label = labels[top_neightbour].label;
                if (top_label != label &&
                    top_label != -1 &&
                    cluster_indices[top_label].indices.size() < tiny_cluster_threshold)
                {
                    to_merge.push_back(top_label);
                }
            }

            for (unsigned merge_label : to_merge) {
                auto &merge_cluster = cluster_indices[merge_label];
                for (int i : merge_cluster.indices) {
                    pcl::PointXYZL &point = pcl_cloud->points[i];
                    int x = i % width;
                    int y = i / width;

                    if (x > large_label.max_x_2d)
                        large_label.max_x_2d = x;
                    if (x < large_label.min_x_2d)
                        large_label.min_x_2d = x;
                    if (y > large_label.max_y_2d)
                        large_label.max_y_2d = y;
                    if (y < large_label.min_y_2d)
                        large_label.min_y_2d = y;

                    if (point.x > large_label.max_x)
                        large_label.max_x = point.x;
                    if (point.x < large_label.min_x)
                        large_label.min_x = point.x;
                    if (point.y > large_label.max_y)
                        large_label.max_y = point.y;
                    if (point.y < large_label.min_y)
                        large_label.min_y = point.y;
                    if (point.z > large_label.max_z)
                        large_label.max_z = point.z;
                    if (point.z < large_label.min_z)
                        large_label.min_z = point.z;

                    cluster.indices.push_back(i);
                }
                merge_cluster.indices.clear();
            }

            if (large_label.max_x_2d != -1 && large_label.max_y_2d != -1) {
                large_clusters.push_back(large_label);
            }
        }
    } else {
        pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud = tracking->downsampled_cloud;

        for (unsigned label = 0; label < cluster_indices.size(); label++)
        {
            auto &cluster = cluster_indices[label];

            if (cluster.indices.size() < large_cluster_threshold)
                continue;

            struct candidate_cluster large_label = { label };

            for (int i : cluster.indices) {
                pcl::PointXYZL &point = pcl_cloud->points[i];
                int x = i % width;
                int y = i / width;

                if (x > large_label.max_x_2d)
                    large_label.max_x_2d = x;
                if (x < large_label.min_x_2d)
                    large_label.min_x_2d = x;
                if (y > large_label.max_y_2d)
                    large_label.max_y_2d = y;
                if (y < large_label.min_y_2d)
                    large_label.min_y_2d = y;

                if (point.x > large_label.max_x)
                    large_label.max_x = point.x;
                if (point.x < large_label.min_x)
                    large_label.min_x = point.x;
                if (point.y > large_label.max_y)
                    large_label.max_y = point.y;
                if (point.y < large_label.min_y)
                    large_label.min_y = point.y;
                if (point.z > large_label.max_z)
                    large_label.max_z = point.z;
                if (point.z < large_label.min_z)
                    large_label.min_z = point.z;
            }

            gm_assert(ctx->log, large_label.max_x_2d != -1 && large_label.max_y_2d != -1,
                      "Spurious, undefined large label bounds");
            large_clusters.push_back(large_label);
        }
    }

    if (ctx->codebook_cluster_merge_large_neighbours)
    {
        tracking_add_debug_text(tracking,
                                "Looking at %d large clusters to possibly merge",
                                (int)large_clusters.size());
        for (int i = 0; i < large_clusters.size(); i++) {
            struct candidate_cluster &large_label = large_clusters[i];
            tracking_add_debug_text(tracking,
                                    "Large Cluster %d (label=%d, %d points): min_x_2d=%d,min_y_2d=%d,max_x_2d=%d,max_y_2d=%d",
                                    i,
                                    large_label.label,
                                    cluster_indices[large_label.label].indices.size(),
                                    large_label.min_x_2d,
                                    large_label.min_y_2d,
                                    large_label.max_x_2d,
                                    large_label.max_y_2d);
        }

        for (auto &current_cluster : large_clusters) {
            auto &current_indices = cluster_indices[current_cluster.label];

            // we may have merged the indices already...
            if (!current_indices.indices.size())
                continue;

            gm_assert(ctx->log,
                      current_indices.indices.size() >= large_cluster_threshold,
                      "Spurious 'large' cluster (label=%d) isn't large (%d points)",
                      current_cluster.label, (int)current_indices.indices.size());
            for (auto &other_cluster : large_clusters) {
                auto &other_indices = cluster_indices[other_cluster.label];

                if (other_cluster.label == current_cluster.label)
                    continue;

                // we may have merged the indices already...
                if (!other_indices.indices.size())
                    continue;

                gm_assert(ctx->log, other_indices.indices.size() >= large_cluster_threshold,
                          "Spurious merge with non-large cluster (label=%d, %d points)",
                          other_cluster.label,
                          (int)other_indices.indices.size());

                int x0 = std::max(current_cluster.min_x_2d, other_cluster.min_x_2d);
                int y0 = std::max(current_cluster.min_y_2d, other_cluster.min_y_2d);
                int x1 = std::min(current_cluster.max_x_2d, other_cluster.max_x_2d);
                int y1 = std::min(current_cluster.max_y_2d, other_cluster.max_y_2d);
                if (x0 <= x1 && y0 <= y1)
                {
                    tracking_add_debug_text(tracking, "Merging label %d (%d points) into %d (%d points)",
                                            other_cluster.label,
                                            (int)other_indices.indices.size(),
                                            current_cluster.label,
                                            (int)current_indices.indices.size());

                   for (int i : other_indices.indices) {
                       current_indices.indices.push_back(i);
                   }
                   other_indices.indices.clear();
                }
            }
        }
    }

    for (auto &large_cluster : large_clusters) {
        if (cluster_indices[large_cluster.label].indices.size())
            state->candidate_clusters.push_back(large_cluster);
    }
}

static void
stage_codebook_cluster_debug_cb(struct gm_tracking_impl *tracking,
                                struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;

    std::vector<struct gm_point_rgba> &debug_cloud = tracking->debug_cloud;
    std::vector<int> &debug_cloud_indices = tracking->debug_cloud_indices;

    pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud = tracking->downsampled_cloud;
    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;

    int nth_large = -1;
    for (unsigned label = 0; label < cluster_indices.size(); label++) {
        auto &cluster = cluster_indices[label];

        if (cluster.indices.size() < ctx->codebook_large_cluster_threshold)
            continue;

        nth_large++;

        if (ctx->debug_codebook_cluster_idx != -1 &&
            ctx->debug_codebook_cluster_idx != nth_large)
        {
            continue;
        }

        for (int i : cluster.indices) {
            pcl::PointXYZL &point = pcl_cloud->points[i];
            struct gm_point_rgba rgba_point;

            rgba_point.x = point.x;
            rgba_point.y = point.y;
            rgba_point.z = point.z;

            png_color *color =
                &default_palette[label % ARRAY_LEN(default_palette)];
            float shade = 1.f - (float)(label / ARRAY_LEN(default_palette)) / 10.f;
            uint8_t r = (uint8_t)(color->red * shade);
            uint8_t g = (uint8_t)(color->green * shade);
            uint8_t b = (uint8_t)(color->blue * shade);
            rgba_point.rgba = r<<24|g<<16|b<<8|0xff;

            debug_cloud.push_back(rgba_point);
            debug_cloud_indices.push_back(i);
        }
    }
}

static void
stage_naive_detect_floor_cb(struct gm_tracking_impl *tracking,
                            struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    unsigned downsampled_cloud_size = tracking->downsampled_cloud->points.size();
    enum tracking_stage debug_stage_id = (enum tracking_stage)state->debug_pipeline_stage;

    // If we've not tracked a human yet, the depth classification may not
    // be reliable - just use a simple clustering technique to find a
    // human and separate them from the floor, then rely on motion detection
    // for subsequent frames.
    int width = (int)tracking->downsampled_cloud->width;
    int height = (int)tracking->downsampled_cloud->height;
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
                tracking->downsampled_cloud->points[idx];
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

    if (state->debug_cloud_mode &&
        (state->debug_pipeline_stage == TRACKING_STAGE_NAIVE_FLOOR ||
         state->debug_pipeline_stage == TRACKING_STAGE_NAIVE_CLUSTER))
    {
        // Draw the lines of focus...
        if (fz != FLT_MAX) {
            float line_x = tracking->downsampled_cloud->points[focal_region[fr_i].idx].x;
            float line_y = tracking->downsampled_cloud->points[focal_region[fr_i].idx].y;
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
    done_mask.resize(downsampled_cloud_size, false);

    pcl::PointXYZL &focus_pt =
        tracking->downsampled_cloud->points[fidx];

    float lowest_point = FLT_MAX;
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
            tracking->downsampled_cloud->points[idx];

        if (fabsf(focus_pt.x - pcl_pt.x) > ctx->cluster_max_width ||
            fabsf(focus_pt.z - pcl_pt.z) > ctx->cluster_max_depth) {
            continue;
        }

        float aligned_y = tracking->ground_cloud->size() ?
            tracking->ground_cloud->points[idx].y :
            tracking->downsampled_cloud->points[idx].y;
        if (aligned_y < lowest_point) {
            lowest_point = aligned_y;
        }

        if (compare_point_depths(tracking->downsampled_cloud,
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
                state->debug_cloud_mode)
            {
                struct gm_point_rgba debug_point;

                debug_point.x = tracking->downsampled_cloud->points[idx].x;
                debug_point.y = tracking->downsampled_cloud->points[idx].y;
                debug_point.z = tracking->downsampled_cloud->points[idx].z;
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
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    float floor_y = state->naive_floor_y;
    float full_size = 2;
    float cell_size = 0.25f;
    float center[] = { 0, floor_y, 2.5f };
    glm::mat4 ground_to_downsampled = glm::inverse(state->to_ground);

    tracking_draw_transformed_grid(tracking, center, full_size, cell_size,
                                   0x00ffffff, ground_to_downsampled);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_naive_cluster_cb(struct gm_tracking_impl *tracking,
                       struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    enum tracking_stage debug_stage_id = (enum tracking_stage)state->debug_pipeline_stage;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    cluster_indices.clear();

    int width = (int)tracking->downsampled_cloud->width;
    int height = (int)tracking->downsampled_cloud->height;

    float lowest_point = state->naive_floor_y;

    std::queue<struct PointCmp> &flood_fill = state->flood_fill;
    std::vector<bool> &done_mask = state->done_mask;

    uint64_t time_threshold = tracking->frame->timestamp -
        (uint64_t)((double)ctx->cluster_from_prev_time_threshold * 1e9);

    if (ctx->cluster_from_prev && ctx->n_tracking &&
        ctx->tracking_history[0]->frame->timestamp >= time_threshold) {
        for (int j = 0; j < ctx->n_joints; ++j) {
            struct gm_joint &joint =
                ctx->tracking_history[0]->skeleton_corrected.joints[j];
            if (!joint.valid) {
                continue;
            }

            // Project joint position into the spaces of the old frame and
            // the current one.
            float ox, oy, nx, ny;
            struct gm_intrinsics *old_intrinsics =
                &ctx->tracking_history[0]->downsampled_intrinsics;
            struct gm_intrinsics *new_intrinsics =
                &tracking->downsampled_intrinsics;

            project_point(&joint.x, old_intrinsics, &ox, &oy);
            project_point(&joint.x, new_intrinsics, &nx, &ny);

            // Flip Y values as we're using this to address a buffer that's
            // organised with the top-left of the image at the zero index.
            oy = old_intrinsics->height - oy;
            ny = new_intrinsics->height - ny;

            // See if the depth has gone past the threshold and if it hasn't,
            // add it as a point to cluster (flood-fill) from.
            int dox = ox;
            int doy = oy;
            if (dox < 0 || dox >= old_intrinsics->width ||
                doy < 0 || doy >= old_intrinsics->height)
            {
                continue;
            }

            int dnx = nx;
            int dny = ny;
            if (dnx < 0 || dnx >= new_intrinsics->width ||
                dny < 0 || dny >= new_intrinsics->height)
            {
                continue;
            }

            float od = ctx->tracking_history[0]->downsampled_cloud->
                points[doy * old_intrinsics->width + dox].z;
            float nd = tracking->downsampled_cloud->
                points[dny * new_intrinsics->width + dnx].z;

            float diff = fabsf(od - nd);
            if (diff <= ctx->cluster_from_prev_dist_threshold) {
                flood_fill.push({ dnx, dny, dnx, dny });
            }
        }
    }
    if (flood_fill.empty()) {
        int fx = state->naive_fx;
        int fy = state->naive_fy;
        flood_fill.push({ fx, fy, fx, fy });
    }
    std::fill(done_mask.begin(), done_mask.end(), false);

    struct candidate_cluster person_cluster = {};
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

        pcl::PointXYZL pcl_pt = tracking->downsampled_cloud->points[idx];
        float aligned_y = tracking->ground_cloud->size() ?
            tracking->ground_cloud->points[idx].y : pcl_pt.y;
        if (aligned_y < lowest_point + ctx->floor_threshold) {
            continue;
        }

        if (compare_point_depths(tracking->downsampled_cloud,
                                 point.x, point.y, point.lx, point.ly,
                                 ctx->cluster_tolerance))
        {
            if (point.x > person_cluster.max_x_2d)
                person_cluster.max_x_2d = point.x;
            if (point.x < person_cluster.min_x_2d)
                person_cluster.min_x_2d = point.x;
            if (point.y > person_cluster.max_y_2d)
                person_cluster.max_y_2d = point.y;
            if (point.y < person_cluster.min_y_2d)
                person_cluster.min_y_2d = point.y;

            if (pcl_pt.x > person_cluster.max_x)
                person_cluster.max_x = pcl_pt.x;
            if (pcl_pt.x < person_cluster.min_x)
                person_cluster.min_x = pcl_pt.x;
            if (pcl_pt.y > person_cluster.max_y)
                person_cluster.max_y = pcl_pt.y;
            if (pcl_pt.y < person_cluster.min_y)
                person_cluster.min_y = pcl_pt.y;
            if (pcl_pt.z > person_cluster.max_z)
                person_cluster.max_z = pcl_pt.z;
            if (pcl_pt.z < person_cluster.min_z)
                person_cluster.min_z = pcl_pt.z;

            person_indices.indices.push_back(idx);
            done_mask[idx] = true;

            flood_fill.push({ point.x - 1, point.y, point.x, point.y });
            flood_fill.push({ point.x + 1, point.y, point.x, point.y });
            flood_fill.push({ point.x, point.y - 1, point.x, point.y });
            flood_fill.push({ point.x, point.y + 1, point.x, point.y });

            // TODO: move outside loop, and instead iterate flood_fill
            // queue when done
            if (debug_stage_id == TRACKING_STAGE_NAIVE_CLUSTER &&
                state->debug_cloud_mode)
            {
                struct gm_point_rgba debug_point;

                debug_point.x = pcl_pt.x;
                debug_point.y = pcl_pt.y;
                debug_point.z = pcl_pt.z;
                debug_point.rgba = 0xffffffff;

                tracking->debug_cloud.push_back(debug_point);
                tracking->debug_cloud_indices.push_back(idx);
            }
        }
    }

    if (!person_indices.indices.empty()) {
        person_cluster.label = cluster_indices.size();
        cluster_indices.push_back(person_indices);
        state->candidate_clusters.push_back(person_cluster);
    }
}

static void
stage_naive_cluster_debug_cb(struct gm_tracking_impl *tracking,
                             struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    // Note: the actual debug cloud is updated as part of
    // stage_naive_cluster_cb above, so we just need the color..
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    float floor_y = state->naive_floor_y;
    float full_size = 2;
    float cell_size = 0.25f;
    float center[] = { 0, floor_y, 2.5f };
    glm::mat4 ground_to_downsampled = glm::inverse(state->to_ground);

    tracking_draw_transformed_grid(tracking, center, full_size, cell_size,
                                   0x00ffffff, ground_to_downsampled);

    center[1] += ctx->floor_threshold;
    tracking_draw_transformed_grid(tracking, center, full_size, cell_size,
                                   0x808080ff, ground_to_downsampled);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_filter_clusters_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    // Check the cluster at least has vaguely human dimensions

    for (auto &candidate : state->candidate_clusters) {
        std::vector<int> &indices = cluster_indices[candidate.label].indices;

        float cluster_width = candidate.max_x - candidate.min_x;
        float cluster_height = candidate.max_y - candidate.min_y;
        float cluster_depth = candidate.max_z - candidate.min_z;

        if (cluster_width < ctx->cluster_min_width ||
            cluster_width > ctx->cluster_max_width ||
            cluster_height < ctx->cluster_min_height ||
            cluster_height > ctx->cluster_max_height ||
            cluster_depth < ctx->cluster_min_depth ||
            cluster_depth > ctx->cluster_max_depth)
        {
            continue;
        }
        gm_info(ctx->log,
                "Person cluster with %d points, (%.2fx%.2fx%.2f)\n",
                (int)indices.size(),
                cluster_width,
                cluster_height,
                cluster_depth);

        person_clusters.push_back(candidate);
    }
}

static void
stage_filter_clusters_debug_cb(struct gm_tracking_impl *tracking,
                              struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;

    for (auto &cluster : state->candidate_clusters) {
        uint32_t color = 0xff0000ff;

        for (auto &person_cluster : state->person_clusters) {
            if (person_cluster.label == cluster.label) {
                color = 0x00ff00ff;
                break;
            }
        }

        std::vector<int> &indices = cluster_indices[cluster.label].indices;
        add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                      tracking->downsampled_cloud,
                                                      indices);

        tracking_draw_line(tracking,
                           cluster.min_x, cluster.min_y, cluster.min_z,
                           cluster.min_x, cluster.max_y, cluster.min_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.max_y, cluster.min_z,
                           cluster.min_x, cluster.max_y, cluster.max_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.max_y, cluster.max_z,
                           cluster.min_x, cluster.min_y, cluster.max_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.min_y, cluster.max_z,
                           cluster.min_x, cluster.min_y, cluster.min_z,
                           color);

        tracking_draw_line(tracking,
                           cluster.max_x, cluster.min_y, cluster.min_z,
                           cluster.max_x, cluster.max_y, cluster.min_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.max_x, cluster.max_y, cluster.min_z,
                           cluster.max_x, cluster.max_y, cluster.max_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.max_x, cluster.max_y, cluster.max_z,
                           cluster.max_x, cluster.min_y, cluster.max_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.max_x, cluster.min_y, cluster.max_z,
                           cluster.max_x, cluster.min_y, cluster.min_z,
                           color);

        tracking_draw_line(tracking,
                           cluster.min_x, cluster.min_y, cluster.min_z,
                           cluster.max_x, cluster.min_y, cluster.min_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.max_y, cluster.min_z,
                           cluster.max_x, cluster.max_y, cluster.min_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.max_y, cluster.max_z,
                           cluster.max_x, cluster.max_y, cluster.max_z,
                           color);
        tracking_draw_line(tracking,
                           cluster.min_x, cluster.min_y, cluster.max_z,
                           cluster.max_x, cluster.min_y, cluster.max_z,
                           color);
    }

    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);
}

static void
stage_crop_cluster_image_cb(struct gm_tracking_impl *tracking,
                            struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<float> &depth_image = ctx->inference_cluster_depth_image;

    float bg_depth = ctx->decision_trees[0]->header.bg_depth;
    gm_assert(ctx->log, !std::isnan(bg_depth),
              "Spurious NaN background value specified in decision tree header");

    pcl::PointCloud<pcl::PointXYZL>::Ptr pcl_cloud = tracking->downsampled_cloud;
    int cloud_width_2d = pcl_cloud->width;
    //int cloud_height_2d = pcl_cloud->width;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->current_person_cluster >= 0,
              "No person cluster selected for cropping");

    auto &cluster = person_clusters[state->current_person_cluster];
    int cluster_width_2d = cluster.max_x_2d - cluster.min_x_2d + 1;
    int cluster_height_2d = cluster.max_y_2d - cluster.min_y_2d + 1;

    size_t img_size = cluster_width_2d * cluster_height_2d;
    depth_image.clear();
    depth_image.resize(img_size, bg_depth);

    std::vector<int> &indices = cluster_indices[cluster.label].indices;
    for (int i : indices) {
        pcl::PointXYZL &point = pcl_cloud->points[i];

        int x = i % cloud_width_2d;
        int y = i / cloud_width_2d;
        int cluster_x = x - cluster.min_x_2d;
        int cluster_y = y - cluster.min_y_2d;

        int doff = cluster_width_2d * cluster_y + cluster_x;
        depth_image[doff] = point.z;
    }
}

static void
stage_crop_cluster_image_debug_cb(struct gm_tracking_impl *tracking,
                                  struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->current_person_cluster >= 0,
              "No person cluster selected");

    struct candidate_cluster &cluster = person_clusters[state->current_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_label_inference_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    //std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->current_person_cluster >= 0,
              "No person cluster selected for cropping");

    auto &cluster = person_clusters[state->current_person_cluster];
    int cluster_width_2d = cluster.max_x_2d - cluster.min_x_2d + 1;
    int cluster_height_2d = cluster.max_y_2d - cluster.min_y_2d + 1;

    ctx->label_probs_back.resize(cluster_width_2d *
                                 cluster_height_2d *
                                 ctx->n_labels);

    infer_labels(ctx->log,
                 ctx->decision_trees,
                 ctx->n_decision_trees,
                 ctx->inference_cluster_depth_image.data(),
                 cluster_width_2d, cluster_height_2d,
                 ctx->label_probs_back.data(),
                 ctx->use_threads,
                 ctx->flip_labels);

    state->done_label_inference = true;
}

static void
stage_label_inference_debug_cb(struct gm_tracking_impl *tracking,
                               struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No person best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_joint_weights_cb(struct gm_tracking_impl *tracking,
                       struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->current_person_cluster >= 0,
              "No person cluster selected");

    auto &cluster = person_clusters[state->current_person_cluster];
    int cluster_width_2d = cluster.max_x_2d - cluster.min_x_2d + 1;
    int cluster_height_2d = cluster.max_y_2d - cluster.min_y_2d + 1;

    ctx->inference_cluster_weights.resize(cluster_width_2d *
                                          cluster_height_2d *
                                          ctx->n_joints);

    joints_inferrer_calc_pixel_weights(ctx->joints_inferrer,
                                       ctx->inference_cluster_depth_image.data(),
                                       ctx->label_probs_back.data(),
                                       cluster_width_2d, cluster_height_2d,
                                       ctx->n_labels,
                                       ctx->inference_cluster_weights.data());
}

static void
stage_joint_weights_debug_cb(struct gm_tracking_impl *tracking,
                             struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_joint_inference_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;
    int seg_res = state->seg_res;

    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->current_person_cluster >= 0,
              "No person cluster selected");

    auto &cluster = person_clusters[state->current_person_cluster];
    int cluster_width_2d = cluster.max_x_2d - cluster.min_x_2d + 1;
    int cluster_height_2d = cluster.max_y_2d - cluster.min_y_2d + 1;

    struct gm_intrinsics downsampled_intrinsics = tracking->depth_camera_intrinsics;
    downsampled_intrinsics.width /= seg_res;
    downsampled_intrinsics.height /= seg_res;
    downsampled_intrinsics.cx /= seg_res;
    downsampled_intrinsics.cy /= seg_res;
    downsampled_intrinsics.fx /= seg_res;
    downsampled_intrinsics.fy /= seg_res;

    if (ctx->fast_clustering) {
        state->joints_candidate =
                joints_inferrer_infer_fast(ctx->joints_inferrer,
                                           &downsampled_intrinsics,
                                           cluster_width_2d, cluster_height_2d,
                                           cluster.min_x_2d, cluster.min_y_2d,
                                           ctx->inference_cluster_depth_image.data(),
                                           ctx->label_probs_back.data(),
                                           ctx->inference_cluster_weights.data(),
                                           ctx->n_labels,
                                           ctx->joint_params->joint_params);
    } else {
        state->joints_candidate =
                joints_inferrer_infer(ctx->joints_inferrer,
                                      &downsampled_intrinsics,
                                      cluster_width_2d, cluster_height_2d,
                                      cluster.min_x_2d, cluster.min_y_2d,
                                      ctx->inference_cluster_depth_image.data(),
                                      ctx->label_probs_back.data(),
                                      ctx->inference_cluster_weights.data(),
                                      ctx->decision_trees[0]->header.bg_depth,
                                      ctx->n_labels,
                                      ctx->joint_params->joint_params);
    }
}

static void
stage_joint_inference_debug_cb(struct gm_tracking_impl *tracking,
                               struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_refine_skeleton_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    copy_inferred_joints_to_skel_except(tracking->skeleton,
                                        tracking->joints,
                                        -1); // no joint to skip copying
    update_bones(ctx, tracking->skeleton);

    tracking->skeleton_corrected = tracking->skeleton;
    refine_skeleton(tracking);
}

static void
stage_refine_skeleton_debug_cb(struct gm_tracking_impl *tracking,
                               struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_sanitize_skeleton_cb(struct gm_tracking_impl *tracking,
                           struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    sanitise_skeleton(ctx, tracking->skeleton_corrected,
                      tracking->frame->timestamp);
}

static void
stage_sanitize_skeleton_debug_cb(struct gm_tracking_impl *tracking,
                                 struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_select_best_person_cloud_debug_cb(struct gm_tracking_impl *tracking,
                                        struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
    std::vector<candidate_cluster> &person_clusters = state->person_clusters;

    gm_assert(ctx->log, state->best_person_cluster >= 0,
              "No best person cluster determined");

    struct candidate_cluster &cluster = person_clusters[state->best_person_cluster];
    std::vector<int> &indices = cluster_indices[cluster.label].indices;

    add_debug_cloud_xyz_from_pcl_xyzl_and_indices(ctx, tracking,
                                                  tracking->downsampled_cloud,
                                                  indices);
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
}

static void
stage_update_codebook_cb(struct gm_tracking_impl *tracking,
                         struct pipeline_scratch_state *state)
{
    if (state->codebook_frozen)
        return;

    struct gm_context *ctx = tracking->ctx;

    uint64_t frame_time = tracking->frame->timestamp;
    uint64_t update_frame_count = state->frame_counter;

    float clear_tracked_threshold = ctx->codebook_clear_tracked_threshold;

    glm::mat4 to_start = state->to_start;
    glm::mat4 to_codebook = state->start_to_codebook;

    std::vector<std::vector<struct seg_codeword>> &seg_codebook =
        *state->seg_codebook;

    struct gm_intrinsics codebook_intrinsics = tracking->downsampled_intrinsics;

    unsigned downsampled_cloud_size = tracking->downsampled_cloud->points.size();
    for (unsigned depth_off = 0; depth_off < downsampled_cloud_size; ++depth_off)
    {
        pcl::PointXYZL point = tracking->downsampled_cloud->points[depth_off];

        if (std::isnan(point.z))
            continue;

        int off = project_point_into_codebook(&point,
                                              to_start,
                                              to_codebook,
                                              &codebook_intrinsics);
        // Falls outside of codebook so we can't classify...
        if (off < 0)
            continue;

        // At this point z has been projected into the coordinate space of
        // the codebook
        float depth = point.z;

        std::vector<struct seg_codeword> &codewords = seg_codebook[off];

        // Delete any codewords that match a tracked point's distance
        //
        // Note: we don't assume the threshold testing can only match one
        // codeword, since the mean distance of a codeword can change
        // and drift over time resulting in codewords becoming arbitrarily
        // close together. (Considering this it may even make sense for us
        // to merge codewords that get too close).
        //
        if (point.label == CODEBOOK_CLASS_TRACKED) {
            for (int i = 0; i < (int)codewords.size(); ) {
                struct seg_codeword &candidate = codewords[i];

                float dist = fabsf(depth - candidate.mean);

                /* Note: we don't typically expect many codewords so don't
                 * expect array removal to really be a significant cost
                 */
                if (dist < clear_tracked_threshold) {
                    codewords.erase(codewords.begin() + i);
                } else
                    i++;
            }
            continue;
        }

        // Look to see if this pixel falls into an existing codeword

        struct seg_codeword *codeword = NULL;
        float best_codeword_distance = FLT_MAX;

        for (int i = 0; i < codewords.size(); i++) {
            struct seg_codeword &candidate = codewords[i];

            /* The codewords are sorted from closest to farthest */
            float dist = fabsf(depth - candidate.mean);
            if (dist < best_codeword_distance) {
                codeword = &candidate;
                best_codeword_distance = dist;
            } else {
                // Any other codewords will be even farther away
                break;
            }
        }
        // NB: ->codebook_bg_threshold = Segmentation bucket threshold
        if (best_codeword_distance > ctx->codebook_bg_threshold)
            codeword = NULL;

        if (!codeword) {
            struct seg_codeword new_codeword = {};
            new_codeword.mean = depth;
            new_codeword.n = 1;
            new_codeword.create_timestamp = frame_time;
            new_codeword.last_update_timestamp = frame_time;
            new_codeword.last_update_frame_count = update_frame_count;
            new_codeword.n_consecutive_update_runs = 0;

            // We insert sorted so that our matching logic can bail as soon
            // as it sees the distance increasing while looking for the
            // nearest match...
            bool inserted = false;
            for (int i = 0; i < codewords.size(); i++) {
                if (codewords[i].mean > depth) {
                    codewords.insert(codewords.begin() + i, new_codeword);
                    inserted = true;
                    break;
                }
            }
            if (!inserted)
                codewords.push_back(new_codeword);
        } else {
            // NB: codeword_mean_n_max = Segmentation max existing mean weight
            // ->n is the number of depth values that the mean is based on
            //
            // We clamp the 'n' value used to update the mean so that we limit
            // the dampening effect that large n values have on the influence
            // of newer depth values...
            float effective_n = (float)std::min(ctx->codeword_mean_n_max, codeword->n);
            codeword->mean = (((effective_n * codeword->mean) + depth) /
                              (effective_n + 1.f));
            codeword->n++;

            /* Here we are counting the breaks in (or start of) consecutive
             * updates to a codeword.
             *
             * E.g. over 10 frames if a point matches the same codeword
             * on frames 1,2 - 4,5,6 - and 8 then there are three consecutive
             * update runs...
             *
             * 'consecutive' is bit of a misnomer since we count 'runs' of one
             * update.
             */
            if (codeword->last_update_timestamp != ctx->last_codebook_update_time)
                codeword->n_consecutive_update_runs++;

            codeword->last_update_timestamp = frame_time;
            codeword->last_update_frame_count = update_frame_count;
        }
    }

    if (!state->paused) {
        ctx->last_codebook_update_time = frame_time;
        ctx->last_codebook_update_frame_counter = update_frame_count;
    }
}

static void
stage_update_history_cb(struct gm_tracking_impl *tracking,
                        struct pipeline_scratch_state *state)
{
    //struct gm_context *ctx = tracking->ctx;

}

static void
stage_update_history_debug_cb(struct gm_tracking_impl *tracking,
                              struct pipeline_scratch_state *state)
{
    struct gm_context *ctx = tracking->ctx;

    add_debug_cloud_xyz_from_pcl_xyzl(ctx, tracking, tracking->downsampled_cloud);

    tracking->debug_cloud_intrinsics = tracking->downsampled_intrinsics;
    colour_debug_cloud(ctx, state, tracking, tracking->downsampled_cloud);

    struct gm_prediction *prediction = NULL;
    struct gm_prediction_impl *prediction_impl = NULL;
    int h1 = -1;
    int h2 = -1;

    struct gm_tracking_impl **tracking_history = NULL;
    int n_tracking = 0;

    if (ctx->debug_predictions) {
        int64_t offset_ns = ctx->debug_prediction_offset * 1e9;
        prediction =
            gm_context_get_prediction(ctx, (int64_t)tracking->frame->timestamp + offset_ns);
        prediction_impl = (struct gm_prediction_impl *)prediction;

        tracking_history = prediction_impl->tracking_history;
        n_tracking = prediction_impl->n_tracking;
        h1 = prediction_impl->h1;
        h2 = prediction_impl->h2;
    } else {
        tracking_history = ctx->tracking_history;
        n_tracking = ctx->n_tracking;
    }

    int n_bones = ctx->n_bones;

    for (int i = 0; i < n_tracking; i++) {
        struct gm_tracking_impl *historic = tracking_history[i];

        float green = 255.0f - 200.0f * ((float)i / (TRACK_FRAMES - 1));
        uint8_t g = green;
        uint32_t rgba = ((uint8_t)g)<<16 | 0xff;

        if (i == h1) {
            rgba = 0x0000ffff;
        } else if (i == h2) {
            rgba = 0xff0000ff;
        }

        for (int b = 0; b < n_bones; b++) {
            struct gm_bone_info &bone_info = ctx->bone_info[b];

            struct gm_joint &joint0 = historic->skeleton_corrected.joints[bone_info.head];
            struct gm_joint &joint1 = historic->skeleton_corrected.joints[bone_info.tail];

            tracking_draw_line(tracking,
                               joint0.x, joint0.y, joint0.z,
                               joint1.x, joint1.y, joint1.z,
                               rgba);
        }
    }

    if (ctx->debug_predictions) {
        struct gm_skeleton *skel = gm_prediction_get_skeleton(prediction);

        for (int b = 0; b < n_bones; b++) {
            struct gm_bone_info &bone_info = ctx->bone_info[b];

            struct gm_joint &joint0 = skel->joints[bone_info.head];
            struct gm_joint &joint1 = skel->joints[bone_info.tail];

            tracking_draw_line(tracking,
                               joint0.x, joint0.y, joint0.z,
                               joint1.x, joint1.y, joint1.z,
                               0xffffffff);
        }

        gm_prediction_unref(prediction);
    }
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
    if (state->debug_cloud_mode &&
        state->debug_pipeline_stage == stage_id &&
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
    } else {
        // so analytics at least see it was run if they only check the
        // duration...
        duration = 1;
    }

    // Note we append to a vector since a stage (such as label inference)
    // may be run multiple times over different candidate clusters.
    stage_data.durations.push_back(duration);
    stage_data.frame_duration_ns += duration;

    stage.total_time_ns += duration;

    gm_info(ctx->log,
            "Stage: %s took %.3f%s",
            stage.name,
            get_duration_ns_print_scale(duration),
            get_duration_ns_print_scale_suffix(duration));

    run_stage_debug(tracking, stage_id, stage_debug_callback, state);
}

static void
context_clear_tracking_history_locked(struct gm_context *ctx)
{
    for (int i = 0; i < ctx->n_tracking; ++i) {
        gm_tracking_unref(&ctx->tracking_history[i]->base);
        ctx->tracking_history[i] = NULL;
    }
    ctx->n_tracking = 0;
    ctx->last_tracking_success_timestamp = 0;
}

/* XXX: this API must either be called with the tracking_swap_mutex held or
 * at some time when synchronization with the tracking thread is guaranteed
 * (such as when destroying the context).
 */
static void
context_clear_tracking_locked(struct gm_context *ctx, bool clear_pools)
{
    if (ctx->latest_paused_tracking) {
        gm_tracking_unref(&ctx->latest_paused_tracking->base);
        ctx->latest_paused_tracking = NULL;
    }

    if (ctx->latest_tracking) {
        gm_tracking_unref(&ctx->latest_tracking->base);
        ctx->latest_tracking = NULL;
    }

    context_clear_tracking_history_locked(ctx);

    if (clear_pools) {
        mem_pool_foreach(ctx->tracking_pool,
                         print_tracking_info_cb,
                         ctx);
        mem_pool_free_resources(ctx->tracking_pool);
        mem_pool_free_resources(ctx->prediction_pool);
    }
}

/* The scratch state itself is stack allocated so we don't need to free
 * the struct itself, but in case we hang any non-RAII allocations off
 * the struct...
 */
static void
pipeline_scratch_state_clear(struct pipeline_scratch_state *state)
{
    /* XXX: everything is RAII managed now so don't need to manually
     * free anything now and we could just remove this function
     */
}

static bool
context_track_skeleton(struct gm_context *ctx,
                       struct gm_tracking_impl *tracking,
                       struct pipeline_scratch_state &state)
{
    tracking->success = false;

    tracking->paused = state.paused;

    // Insulate the full tracking pipeline from any async property changes
    state.seg_res = ctx->seg_res;
    bool motion_detection = ctx->motion_detection;
    bool naive_seg_fallback = ctx->naive_seg_fallback;

    for (int i = 0; i < tracking->stage_data.size(); i++) {
        tracking->stage_data[i].frame_duration_ns = 0;
        tracking->stage_data[i].durations.clear();
    }

    /* Especially for a debug build if stages take a long time to process
     * it's possible that changes to this state could result in running
     * multiple incompatible debug stages if we didn't snapshot at the
     * start...
     */
    state.debug_pipeline_stage = ctx->debug_pipeline_stage;
    state.debug_cloud_mode = ctx->debug_cloud_mode;

    if (state.debug_cloud_mode) {
        gm_debug(ctx->log, "Clearing debug visualization state");
        tracking->debug_cloud.resize(0);
        memset(&tracking->debug_cloud_intrinsics, 0,
               sizeof(tracking->debug_cloud_intrinsics));
        tracking->debug_cloud_indices.resize(0);
        tracking->debug_lines.resize(0);
    }

    for (auto string : tracking->debug_text) {
        free(string);
    }
    tracking->debug_text.resize(0);

    /* Discontinuities will e.g. happen when a recording loops or if we jump
     * frames in a recording.
     */
    if (tracking->frame->discontinuity) {
        gm_debug(ctx->log, "Wiping codebook (frame discontinuity)");

        /* Note: this isn't the same as resetting the codebook
         * later in cases that the pose has changed since we don't
         * affect the codebook_pose here.
         */
        unsigned int codebook_size = ctx->seg_codebook.size();
        if (!state.codebook_frozen) {
            ctx->seg_codebook.clear();
            ctx->seg_codebook.resize(codebook_size);
        }

        gm_debug(ctx->log, "Clearing tracking history (frame discontinuity)");

        pthread_mutex_lock(&ctx->tracking_swap_mutex);
        context_clear_tracking_history_locked(ctx);
        pthread_mutex_unlock(&ctx->tracking_swap_mutex);
    }

    float nan = std::numeric_limits<float>::quiet_NaN();
    pcl::PointXYZL invalid_pt;
    invalid_pt.x = invalid_pt.y = invalid_pt.z = nan;
    invalid_pt.label = -1;

    // copy + rotate the latest depth buffer
    run_stage(tracking,
              TRACKING_STAGE_START,
              stage_start_cb,
              stage_start_debug_cb,
              &state);

    run_stage(tracking,
              TRACKING_STAGE_NEAR_FAR_CULL_AND_INFILL,
              stage_near_far_cull_and_infill_cb,
              stage_near_far_cull_and_infill_debug_cb,
              &state);

    run_stage(tracking,
              TRACKING_STAGE_DOWNSAMPLED,
              stage_downsample_cb,
              stage_downsample_debug_cb,
              &state);
    unsigned downsampled_cloud_size = tracking->downsampled_cloud->points.size();

    /* Note: we also run this stage when ctx->delete_edges == false
     * if selected for debugging, just so we can visualize what
     * points would get deleted...
     */
    if (ctx->delete_edges ||
        (state.debug_cloud_mode &&
         state.debug_pipeline_stage == TRACKING_STAGE_EDGE_DETECT))
    {
        run_stage(tracking,
                  TRACKING_STAGE_EDGE_DETECT,
                  stage_edge_detect_cb,
                  stage_edge_detect_debug_cb,
                  &state);
    }

    bool reset_codebook = false;
    if (ctx->seg_codebook.size() != downsampled_cloud_size ||
        (ctx->codebook_pose.type != tracking->frame->pose.type))
    {
        gm_debug(ctx->log, "Resetting codebook (pose type changed or inconsistent size)");
        reset_codebook = true;
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

        while (angle > 180.f)
            angle -= 360.f;

        float distance = glm::distance(
            glm::vec3(ctx->codebook_pose.translation[0],
                      ctx->codebook_pose.translation[1],
                      ctx->codebook_pose.translation[2]),
            glm::vec3(tracking->frame->pose.translation[0],
                      tracking->frame->pose.translation[1],
                      tracking->frame->pose.translation[2]));

        gm_debug(ctx->log, "Pose Angle: %.2f, "
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
            gm_debug(ctx->log, "Resetting codebook (moved too much)");
            reset_codebook = true;
        }
    }

    state.to_start = glm::mat4(1.0);
    state.to_start_valid = false;
    state.to_ground = glm::mat4(1.0);
    state.to_ground_valid = false;
    switch (tracking->frame->pose.type) {
    case GM_POSE_INVALID:
        if (tracking->frame->gravity_valid) {
            struct gm_frame *frame = tracking->frame;

            /* Note this gravity vector should already have been rotated,
             * considering any 90degree rotations needed to account for the
             * physical rotation of the camera (frame->camera_rotation)...
             */
            float *gravity = frame->gravity;

            glm::vec3 down(0.f, -1.f, 0.f);
            glm::vec3 norm_gravity = glm::normalize(
                glm::vec3(gravity[0], gravity[1], gravity[2]));
            glm::vec3 axis = glm::normalize(glm::cross(norm_gravity, down));
            float angle = acosf(glm::dot(norm_gravity, down));
            glm::quat orientation = glm::angleAxis(angle, axis);

            struct gm_pose frame_pose = {};
            frame_pose.type = GM_POSE_TO_GROUND;
            frame_pose.orientation[0] = orientation.x;
            frame_pose.orientation[1] = orientation.y;
            frame_pose.orientation[2] = orientation.z;
            frame_pose.orientation[3] = orientation.w;

            state.to_ground = pose_to_matrix(tracking->frame->pose);
            state.to_ground_valid = true;
        }
        break;
    case GM_POSE_TO_START:
        state.to_start = pose_to_matrix(tracking->frame->pose);
        state.to_start_valid = true;
        state.to_ground = state.to_start;
        state.to_ground_valid = true;
        break;
    case GM_POSE_TO_GROUND:
        state.to_ground = pose_to_matrix(tracking->frame->pose);
        state.to_ground_valid = true;
        break;
    }

    run_stage(tracking,
              TRACKING_STAGE_GROUND_SPACE,
              stage_ground_project_cb,
              stage_ground_project_debug_cb,
              &state);

    if (motion_detection) {

        if (state.codebook_frozen)
            reset_codebook = false;

        if (reset_codebook) {
            if (state.paused)
                state.seg_codebook = &ctx->pause_frame_seg_codebook;
            else
                state.seg_codebook = &ctx->seg_codebook;

            state.seg_codebook->clear();
            state.seg_codebook->resize(downsampled_cloud_size);
            state.codebook_pose = tracking->frame->pose;
            state.start_to_codebook = glm::inverse(state.to_start);

            // Don't modify context state for paused frames
            if (!state.paused) {
                ctx->codebook_pose = state.codebook_pose;
                ctx->start_to_codebook = state.start_to_codebook;
            }

            if (tracking->frame->pose.type != GM_POSE_TO_START)
                gm_debug(ctx->log, "No tracking pose");
        } else {
            if (state.paused) {
                // Relying on C++ assignment operator magic to do a deep copy of
                // the codebook here...
                ctx->pause_frame_seg_codebook = ctx->seg_codebook;
                state.seg_codebook = &ctx->pause_frame_seg_codebook;
            } else
                state.seg_codebook = &ctx->seg_codebook;

            state.codebook_pose = ctx->codebook_pose;
            state.start_to_codebook = ctx->start_to_codebook;

            run_stage(tracking,
                      TRACKING_STAGE_CODEBOOK_RETIRE_WORDS,
                      stage_codebook_retire_cb,
                      stage_codebook_retire_debug_cb,
                      &state);

            run_stage(tracking,
                      TRACKING_STAGE_CODEBOOK_RESOLVE_BACKGROUND,
                      stage_codebook_resolve_background_cb,
                      stage_codebook_resolve_background_debug_cb,
                      &state);

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

            run_stage(tracking,
                      TRACKING_STAGE_CODEBOOK_CLUSTER,
                      stage_codebook_cluster_cb,
                      stage_codebook_cluster_debug_cb,
                      &state);

            run_stage(tracking,
                      TRACKING_STAGE_FILTER_CLUSTERS,
                      stage_filter_clusters_cb,
                      stage_filter_clusters_debug_cb,
                      &state);
        }
    }

    if (naive_seg_fallback && state.person_clusters.size() == 0) {
        gm_debug(ctx->log, "Running naive segmentation");

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

        run_stage(tracking,
                  TRACKING_STAGE_FILTER_CLUSTERS,
                  stage_filter_clusters_cb,
                  stage_filter_clusters_debug_cb,
                  &state);
    }

    if (state.person_clusters.size() == 0) {
        if (motion_detection) {
            run_stage(tracking,
                      TRACKING_STAGE_UPDATE_CODEBOOK,
                      stage_update_codebook_cb,
                      NULL,
                      &state);
        }
        gm_info(ctx->log, "Give up tracking frame: Could not find a person cluster");
        return false;
    }


    std::vector<candidate_cluster> &person_clusters = state.person_clusters;
    state.best_person_cluster = -1;
    state.current_person_cluster = -1;
    state.confidence = 0.f;

    tracking->label_probs.clear();
    tracking->label_probs_width = 0;
    tracking->label_probs_height = 0;

    gm_assert(ctx->log, state.person_clusters.size() > 0,
              "Spurious empty array of candidate person clusters");

    for (state.current_person_cluster = 0;
         state.current_person_cluster < state.person_clusters.size();
         state.current_person_cluster++)
    {
        run_stage(tracking,
                  TRACKING_STAGE_CROP_CLUSTER_IMAGE,
                  stage_crop_cluster_image_cb,
                  stage_crop_cluster_image_debug_cb,
                  &state);

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

        gm_assert(ctx->log, state.joints_candidate == NULL,
                  "Spurious non-NULL candidate joints before joint inference");
        run_stage(tracking,
                  TRACKING_STAGE_JOINT_INFERENCE,
                  stage_joint_inference_cb,
                  NULL,
                  &state);

        gm_assert(ctx->log,
                  state.joints_candidate->n_joints == ctx->n_joints,
                  "ctx->n_joints != joints_candidate->n_joints");

        // Calculate cumulative confidence of the joint inference of this cloud
        float confidence = 0.f;
        for (int j = 0; j < ctx->n_joints; ++j) {
            LList *joints = state.joints_candidate->joints[j];
            if (joints) {
                Joint *joint = (Joint *)joints->data;
                confidence += joint->confidence;
            }
        }

        // If this skeleton has higher confidence than the last, keep it
        if (state.current_person_cluster == 0 || confidence > state.confidence)
        {
            auto &cluster = person_clusters[state.current_person_cluster];
            int cluster_width_2d = cluster.max_x_2d - cluster.min_x_2d + 1;
            int cluster_height_2d = cluster.max_y_2d - cluster.min_y_2d + 1;

            std::swap(tracking->label_probs, ctx->label_probs_back);
            tracking->label_probs_width = cluster_width_2d;
            tracking->label_probs_height = cluster_height_2d;
            std::swap(tracking->joints, state.joints_candidate);
            state.best_person_cluster = state.current_person_cluster;
            state.confidence = confidence;
        }

        if (state.joints_candidate) {
            joints_inferrer_free_joints(ctx->joints_inferrer,
                                        state.joints_candidate);
            state.joints_candidate = NULL;
        }
    }

    state.current_person_cluster = -1;

    gm_assert(ctx->log, state.best_person_cluster >= 0,
              "Failed to select best person cluster");

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
    // we've processed all candidates before updating the debug
    // state...
    run_stage_debug(tracking,
                    TRACKING_STAGE_LABEL_INFERENCE,
                    stage_label_inference_debug_cb,
                    &state);
    run_stage_debug(tracking,
                    TRACKING_STAGE_JOINT_WEIGHTS,
                    stage_joint_weights_debug_cb,
                    &state);
    run_stage_debug(tracking,
                    TRACKING_STAGE_JOINT_INFERENCE,
                    stage_joint_inference_debug_cb,
                    &state);

    run_stage(tracking,
              TRACKING_STAGE_REFINE_SKELETON,
              stage_refine_skeleton_cb,
              stage_refine_skeleton_debug_cb,
              &state);

    tracking_add_debug_text(tracking, "Skeleton confidence: %f", state.confidence);

    // TODO: We just take the most confident skeleton above, but we should
    //       probably establish some thresholds and spit out multiple
    //       skeletons.
    run_stage(tracking,
              TRACKING_STAGE_SANITIZE_SKELETON,
              stage_sanitize_skeleton_cb,
              NULL,
              &state);

    float skel_dist = calc_skeleton_distance(ctx, &tracking->skeleton_corrected);
    tracking_add_debug_text(tracking, "Skeleton distance: %f", skel_dist);

    bool valid_skeleton = true;
    if (ctx->skeleton_validation) {
        valid_skeleton = (state.confidence >= ctx->skeleton_min_confidence &&
                          skel_dist <= ctx->skeleton_max_distance);
    }

#warning "XXX: Setting codebook labels by mapping inference points to downsampled points (potentially different resolutions) seems like a bad idea"
    if (motion_detection) {
        std::vector<pcl::PointIndices> &cluster_indices = tracking->cluster_indices;
        std::vector<candidate_cluster> &person_clusters = state.person_clusters;

        for (int i = 0; i < state.person_clusters.size(); i++) {
            struct candidate_cluster &cluster = person_clusters[i];
            std::vector<int> &indices = cluster_indices[cluster.label].indices;

            if (valid_skeleton && i == state.best_person_cluster) {
                for (auto &idx : indices) {
                    tracking->downsampled_cloud->points[idx].label =
                        CODEBOOK_CLASS_TRACKED;
                }
            } else {
                for (auto &idx : indices) {
                    tracking->downsampled_cloud->points[idx].label =
                        CODEBOOK_CLASS_FAILED_CANDIDATE;
                }
            }
        }
    }

    if (!valid_skeleton) {
        gm_info(ctx->log, "Give up tracking frame: Skeleton validation for best candidate failed");
    }

    if (motion_detection) {
        run_stage(tracking,
                  TRACKING_STAGE_UPDATE_CODEBOOK,
                  stage_update_codebook_cb,
                  NULL,
                  &state);
    }

    tracking->success = valid_skeleton;

    return tracking->success;
}

static void
context_detect_faces(struct gm_context *ctx, struct gm_tracking_impl *tracking)
{
    if (!tracking->face_detect_buf) {
        gm_info(ctx->log, "NULL tracking->face_detect_buf");
        return;
    }

    uint64_t start, end, duration_ns;
    std::vector<dlib::rectangle> face_rects(0);
    glimpse::wrapped_image<unsigned char> grey_img;
    dlib::rectangle buf_rect(tracking->face_detect_buf_width, tracking->face_detect_buf_height);

    gm_info(ctx->log, "New camera frame to process");

    if (ctx->last_faces.size()) {

        gm_info(ctx->log, "Searching %d region[s] for faces", (int)ctx->last_faces.size());

        for (dlib::rectangle &rect : ctx->last_faces) {

            rect = dlib::grow_rect(static_cast<dlib::rectangle&>(rect), (long)((float)rect.width() * 0.4f));
            rect.intersect(buf_rect);

            grey_img.wrap(rect.width(),
                          rect.height(),
                          tracking->face_detect_buf_width, //stride
                          static_cast<unsigned char *>(tracking->face_detect_buf +
                                                       rect.top() * tracking->face_detect_buf_width +
                                                       rect.left()));
            gm_info(ctx->log, "Starting constrained face detection with %dx%d sub image",
                    (int)rect.width(), (int)rect.height());
            start = get_time();
            std::vector<dlib::rectangle> dets = ctx->detector(grey_img);
            end = get_time();
            duration_ns = end - start;
            gm_info(ctx->log, "Number of detected faces = %d, %.3f%s",
                    (int)dets.size(),
                    get_duration_ns_print_scale(duration_ns),
                    get_duration_ns_print_scale_suffix(duration_ns));

            if (dets.size() != 1) {
                gm_error(ctx->log, "Constrained search was expected to find exactly one face - fallback");
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
        gm_info(ctx->log, "Starting face detection with %dx%d image",
                (int)tracking->face_detect_buf_width,
                (int)tracking->face_detect_buf_height);
        start = get_time();
        face_rects = ctx->detector(grey_img);
        end = get_time();
        duration_ns = end - start;
        gm_info(ctx->log, "Number of detected faces = %d, %.3f%s",
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

        gm_info(ctx->log, "Detected %d face %d features in %.3f%s",
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
        gm_error(ctx->log, "Copied face detect buffer for debug overlay in %.3f%s",
                 get_duration_ns_print_scale(duration_ns),
                 get_duration_ns_print_scale_suffix(duration_ns));
    }
#endif
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
    gm_info(ctx->log, "Started resizing frame");
    start = get_time();
    dlib::resize_image(orig_grey_img, grey_1_2_img,
                       dlib::interpolate_bilinear());
    end = get_time();
    duration_ns = end - start;
    gm_info(ctx->log, "Frame scaled to 1/2 size on CPU in %.3f%s",
            get_duration_ns_print_scale(duration_ns),
            get_duration_ns_print_scale_suffix(duration_ns));

#ifdef DOWNSAMPLE_1_4
    start = get_time();
    dlib::resize_image(grey_1_2_img, grey_1_4_img,
                       dlib::interpolate_bilinear());
    end = get_time();
    duration_ns = end - start;

    gm_info(ctx->log, "Frame scaled to 1/4 size on CPU in %.3f%s",
            get_duration_ns_print_scale(duration_ns),
            get_duration_ns_print_scale_suffix(duration_ns));
#endif // DOWNSAMPLE_1_4
#endif // DOWNSAMPLE_1_2


#endif // !DOWNSAMPLE_ON_GPU
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

    //gm_info(ctx->log, "Dropped all but the first (front-facing HOG) from the DLib face detector");
    //ctx->detector.w.resize(1);

    //gm_info(ctx->log, "Detector debug %p", &ctx->detector.scanner);

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

        gm_info(ctx->log, "Waiting for new frame to start tracking\n");
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

        struct pipeline_scratch_state state = {};
        state.paused = frame->paused;
        state.codebook_frozen = ctx->codebook_frozen;
        state.frame_counter = ctx->frame_counter++;

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

        context_detect_faces(ctx, tracking);
#endif
        end = get_time();
        duration = end - start;
        gm_debug(ctx->log, "Tracking preparation took %.3f%s",
                 get_duration_ns_print_scale(duration),
                 get_duration_ns_print_scale_suffix(duration));

        start = get_time();

        bool tracked = context_track_skeleton(ctx, tracking, state);

        end = get_time();
        duration = end - start;
        gm_debug(ctx->log, "Skeletal tracking took %.3f%s",
                 get_duration_ns_print_scale(duration),
                 get_duration_ns_print_scale_suffix(duration));

        if (tracked) {
            gm_info(ctx->log, "Successfully tracked frame");
        } else {
            gm_info(ctx->log, "Failed to track frame");
        }

        pthread_mutex_lock(&ctx->tracking_swap_mutex);

        if (tracked && tracking->paused == false) {
            if (ctx->n_tracking) {
                gm_assert(ctx->log,
                          tracking->frame->timestamp > ctx->tracking_history[0]->frame->timestamp,
                          "Tracking can't be added to history with old timestamp");
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

            ctx->last_tracking_success_timestamp = frame->timestamp;
        }

        /* Hold onto the latest tracking regardless of whether it was
         * successful so that a user can still access all the information
         * related to tracking.
         *
         * We don't want to touch ctx->latest_tracking if we've processed
         * a paused frame since ctx->latest_tracking affects the behaviour
         * of tracking which we don't want while we may be repeatedly
         * re-processing the same frame over and over.
         */
        if (tracking->paused) {
            if (ctx->latest_paused_tracking)
                gm_tracking_unref(&ctx->latest_paused_tracking->base);
            ctx->latest_paused_tracking = tracking;
        } else {
            if (ctx->latest_paused_tracking)
                gm_tracking_unref(&ctx->latest_paused_tracking->base);
            ctx->latest_paused_tracking = NULL;

            if (ctx->latest_tracking)
                gm_tracking_unref(&ctx->latest_tracking->base);
            ctx->latest_tracking = tracking;
        }

        pthread_mutex_unlock(&ctx->tracking_swap_mutex);

        run_stage_debug(tracking,
                        TRACKING_STAGE_UPDATE_HISTORY,
                        stage_update_history_debug_cb,
                        &state);

        pipeline_scratch_state_clear(&state);

        notify_tracking(ctx);

        gm_debug(ctx->log, "Requesting new frame for skeletal tracking");
        /* We throttle frame acquisition according to our tracking rate... */
        request_frame(ctx);


        /* Maintain running statistics about pipeline stage timings
         */
        pthread_mutex_lock(&ctx->aggregate_metrics_mutex);

        tracking->duration_ns = duration;
        ctx->total_tracking_duration += duration;
        ctx->n_frames++;
        for (int i = 0; i < tracking->stage_data.size(); i++) {
            const int max_hist_len = 30;

            struct gm_pipeline_stage &stage = ctx->stages[i];

            uint64_t frame_duration_ns = 0;

            for (int invocation_duration_ns : tracking->stage_data[i].durations)
            {
                frame_duration_ns += invocation_duration_ns;

                stage.n_invocations++;

                if (stage.invocation_duration_hist.size() < max_hist_len) {
                    stage.invocation_duration_hist.push_back(invocation_duration_ns);
                } else {
                    int head = stage.invocation_duration_hist_head;
                    stage.invocation_duration_hist[head] = invocation_duration_ns;
                    stage.invocation_duration_hist_head++;
                    stage.invocation_duration_hist_head %= max_hist_len;
                }
            }

            stage.n_frames++;
            if (stage.frame_duration_hist.size() < max_hist_len) {
                stage.frame_duration_hist.push_back(frame_duration_ns);
            } else {
                int head = stage.frame_duration_hist_head;
                stage.frame_duration_hist[head] = frame_duration_ns;
                stage.frame_duration_hist_head++;
                stage.frame_duration_hist_head %= max_hist_len;
            }
        }

        pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);
    }

    return NULL;
}

static int
start_tracking_thread(struct gm_context *ctx, char **err)
{
    /* XXX: maybe make it an explicit, public api to start running detection
     */
    int ret = pthread_create(&ctx->tracking_thread,
                             nullptr, /* default attributes */
                             detector_thread_cb,
                             ctx);

#ifdef __linux__
    if (ret == 0) {
        pthread_setname_np(ctx->tracking_thread, "Glimpse Track");
    }
#endif

    return ret;
}

static void
stop_tracking_thread(struct gm_context *ctx)
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

    if (ctx->tracking_thread) {
        void *tracking_retval = NULL;
        int ret = pthread_join(ctx->tracking_thread, &tracking_retval);
        if (ret < 0) {
            gm_error(ctx->log, "Failed waiting for tracking thread to complete: %s",
                     strerror(ret));
        } else {
            ctx->tracking_thread = 0;
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
     * stop_tracking_thread() is called.
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

    stop_tracking_thread(ctx);

    /* XXX: we don't need to hold the tracking_swap_mutex here because we've
     * stopped the tracking thread...
     */
    context_clear_tracking_locked(ctx,
                                  true); // and clear tracking/prediction pools

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

    if (ctx->joints_inferrer) {
        joints_inferrer_destroy(ctx->joints_inferrer);
        ctx->joints_inferrer = NULL;
    }

    if (ctx->joint_params) {
        jip_free(ctx->joint_params);
        ctx->joint_params = NULL;
    }

    if (ctx->joint_map) {
        json_value_free(ctx->joint_map);
        ctx->joint_map = NULL;
        ctx->joint_blender_names.resize(0);
        ctx->joint_names.resize(0);
        ctx->joint_semantics.resize(0);
    }

    if (ctx->label_map) {
        json_value_free(ctx->label_map);
        ctx->label_map = NULL;
    }

    delete ctx;
}

static bool
parse_bone_info(struct gm_context *ctx,
                JSON_Value *bone_value,
                int bone_parent_idx,
                char **err)
{
    JSON_Object *bone = json_object(bone_value);
    const char *head_name = json_object_get_string(bone, "head");
    const char *tail_name = json_object_get_string(bone, "tail");

    struct gm_bone_info info = {};
    info.idx = ctx->bone_info.size();
    info.parent = bone_parent_idx;
    info.head = -1;
    info.tail = -1;

    int n_joints = ctx->n_joints;
    for (int i = 0; i < n_joints; i++) {
        if (strcmp(ctx->joint_blender_names[i], head_name) == 0)
            info.head = i;
        else if (strcmp(ctx->joint_blender_names[i], tail_name) == 0)
            info.tail = i;

        if (info.head > 0 && info.tail > 0)
            break;
    }
    if (info.head < 0 || info.tail < 0) {
        gm_throw(ctx->log, err,
                 "Failed to resolve head (%s) and tail (%s) joints for bone",
                 head_name, tail_name);
        return false;
    }

    if (bone_parent_idx >= 0) {
        struct gm_bone_info &parent_info = ctx->bone_info[bone_parent_idx];

        gm_assert(ctx->log, parent_info.n_children < MAX_BONE_CHILDREN,
                  "Can't add more than %d children to a bone",
                  MAX_BONE_CHILDREN);

        parent_info.children[parent_info.n_children++] = info.idx;
    }

    info.min_length = (float)json_object_get_number(bone, "min_length");
    info.mean_length = (float)json_object_get_number(bone, "mean_length");
    info.max_length = (float)json_object_get_number(bone, "max_length");

    if (json_object_has_value(bone, "rotation_constraint")) {
        info.has_rotation_constraint = true;
        JSON_Object *constraint =
            json_object_get_object(bone, "rotation_constraint");
        JSON_Object *rotation = json_object_get_object(constraint, "rotation");
        info.avg_rotation = glm::quat(
            json_object_get_number(rotation, "w"),
            json_object_get_number(rotation, "x"),
            json_object_get_number(rotation, "y"),
            json_object_get_number(rotation, "z"));
        info.max_radius = json_object_get_number(constraint, "radius");
    }

    ctx->bone_info.push_back(info);

    JSON_Array *children = json_object_get_array(bone, "children");
    int n_children = json_array_get_count(children);
    for (int i = 0; i < n_children; i++) {
        JSON_Value *child = json_array_get_value(children, i);
        if (!parse_bone_info(ctx, child, info.idx, err))
            return false;
    }

    return true;
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
    pthread_mutex_init(&ctx->aggregate_metrics_mutex, NULL);

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

    int ret = start_tracking_thread(ctx, err);
    if (ret != 0) {
        gm_throw(logger, err,
                 "Failed to start tracking thread: %s", strerror(ret));
        gm_context_destroy(ctx);
    }

    /* We *optionally* open a label map so that we can describe an _ENUM
     * property with appropriate label names, but if the file is missing
     * the enum will simply create names like "Label 0", "Label 1"...
     */
    char *open_err = NULL;
    struct gm_asset *label_map_asset = gm_asset_open(logger,
                                                     "label-map.json",
                                                     GM_ASSET_MODE_BUFFER,
                                                     &open_err);
    if (label_map_asset) {
        const void *buf = gm_asset_get_buffer(label_map_asset);
        unsigned len = gm_asset_get_length(label_map_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length and expects a NUL terminated string...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';

        ctx->label_map = json_parse_string(js_string);

        uint8_t ignored_mapping[256]; // we only care about the names
        if (!gm_data_parse_label_map(logger, ctx->label_map,
                                     ctx->n_labels, ignored_mapping, &open_err))
        {
            xfree(open_err);
            open_err = NULL;
            json_value_free(ctx->label_map);
            ctx->label_map = NULL;
        }

        xfree(js_string);
        gm_asset_close(label_map_asset);
    } else {
        xfree(open_err);
        open_err = NULL;
    }

    if (!ctx->label_map) {
        /* It's not considered an error to be missing this */
        gm_info(logger, "No label map asset opened, so can't refer to names of labels in properties/debugging");
    }


    ctx->joint_map = NULL;
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

    } else {
        gm_throw(logger, err, "Failed to open joint-map.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }

    int n_joints = json_array_get_count(json_array(ctx->joint_map));
    if (n_joints == 0) {
        gm_throw(logger, err, "Joint map contains no joints");
        gm_context_destroy(ctx);
        return NULL;
    }

    ctx->n_joints = n_joints;
    ctx->joint_blender_names.resize(n_joints);
    ctx->joint_names.resize(n_joints);
    ctx->joint_semantics.resize(n_joints);
    for (int i = 0; i < n_joints; i++) {
        const char *blender_name = json_object_get_string(
            json_array_get_object(json_array(ctx->joint_map), i), "joint");
        ctx->joint_blender_names[i] = blender_name;
        ctx->joint_names[i] = "Unknown";
        ctx->joint_semantics[i] = GM_JOINT_UNKNOWN;

        /* FIXME: these mappings should probably come from the joint map
         * directly
         */
        struct {
            const char *blender_name;
            const char *name;
            enum gm_joint_semantic semantic;
        } blender_name_to_info_map[] = {
            { "head.tail", "Head", GM_JOINT_HEAD },
            { "neck_01.head", "Neck", GM_JOINT_NECK },
            { "upperarm_l.head", "Left Shoulder", GM_JOINT_LEFT_SHOULDER },
            { "upperarm_r.head", "Right Shoulder", GM_JOINT_RIGHT_SHOULDER },
            { "lowerarm_l.head", "Left Elbow", GM_JOINT_LEFT_ELBOW },
            { "lowerarm_l.tail", "Left Wrist", GM_JOINT_LEFT_WRIST },
            { "lowerarm_r.head", "Right Elbow", GM_JOINT_RIGHT_ELBOW },
            { "lowerarm_r.tail", "Right Wrist", GM_JOINT_RIGHT_WRIST },
            { "thigh_l.head", "Left Hip", GM_JOINT_LEFT_HIP },
            { "thigh_l.tail", "Left Knee", GM_JOINT_LEFT_KNEE },
            { "thigh_r.head", "Right Hip", GM_JOINT_RIGHT_HIP },
            { "thigh_r.tail", "Right Knee", GM_JOINT_RIGHT_KNEE },
            { "foot_l.head", "Left Ankle", GM_JOINT_LEFT_ANKLE },
            { "foot_r.head", "Right Ankle", GM_JOINT_RIGHT_ANKLE },
            { NULL, "Unknown", GM_JOINT_UNKNOWN }, // NULL sentinel
        };

        for (int j = 0; blender_name_to_info_map[j].name; j++) {
            if (strcmp(blender_name, blender_name_to_info_map[j].blender_name) == 0) {
                ctx->joint_names[i] = blender_name_to_info_map[j].name;
                ctx->joint_semantics[i] = blender_name_to_info_map[j].semantic;
                break;
            }
        }

        /* Maybe allow this later for some kind of backwards compatibility
         * but for now it most likely implies a mistake has been made...
         */
        gm_assert(ctx->log, ctx->joint_semantics[i] != GM_JOINT_UNKNOWN,
                  "Unknown joint semantic");
    }

    struct gm_asset *bone_map_asset = gm_asset_open(logger,
                                                     "bone-map.json",
                                                     GM_ASSET_MODE_BUFFER,
                                                     &open_err);
    if (bone_map_asset) {
        const void *buf = gm_asset_get_buffer(bone_map_asset);
        unsigned len = gm_asset_get_length(bone_map_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length and expects a NUL terminated string...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';

        JSON_Value *bone_map = json_parse_string(js_string);

        xfree(js_string);
        js_string = NULL;

        gm_asset_close(bone_map_asset);
        bone_map_asset = NULL;

        if (!bone_map) {
            gm_throw(logger, err, "Failed to parse bone map json\n");
            gm_context_destroy(ctx);
            return NULL;
        }

        if (!parse_bone_info(ctx,
                             bone_map,
                             -1, // no parent for root bone
                             err))
        {
            gm_context_destroy(ctx);
            return NULL;
        }

        ctx->n_bones = ctx->bone_info.size();
    } else {
        gm_throw(logger, err, "Failed to open bone-map.json: %s", open_err);
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

    ctx->joints_inferrer = joints_inferrer_new(ctx->log,
                                               ctx->joint_map, err);
    if (!ctx->joints_inferrer) {
        gm_context_destroy(ctx);
        return NULL;
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

    ctx->delete_edges = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "delete_edges";
    prop.desc = "Detect edges and invalidate edge points";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->delete_edges;
    ctx->properties.push_back(prop);

    ctx->motion_detection = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "motion_detection";
    prop.desc = "Enable motion-based human segmentation";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->motion_detection;
    ctx->properties.push_back(prop);

    ctx->naive_seg_fallback = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "naive_seg_fallback";
    prop.desc = "Enable a naive segmentation fallback, based on assuming the camera is pointing directly at a person";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->naive_seg_fallback;
    ctx->properties.push_back(prop);

    ctx->joint_refinement = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_refinement";
    prop.desc = "Favour less confident joint predictions that conform "
                "to previous tracked joint positions better";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->joint_refinement;
    ctx->properties.push_back(prop);

    ctx->max_joint_refinement_delta = 0.2f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_joint_refinement_delta";
    prop.desc = "The maximum time between joint refinement comparisons, "
                "in seconds";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->max_joint_refinement_delta;
    prop.float_state.min = 0.f;
    prop.float_state.max = 1.f;
    ctx->properties.push_back(prop);

    ctx->prediction_dampen_large_deltas = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "prediction_dampen_large_deltas";
    prop.desc = "Should we dampen predictions that deviate too far from the known data we interpolating from";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->prediction_dampen_large_deltas;
    ctx->properties.push_back(prop);

    ctx->prediction_interpolate_angles = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "prediction_interpolate_angles";
    prop.desc = "Interpolate angles of bones (not just positions of joints) when predicting skeletons";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->prediction_interpolate_angles;
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

    ctx->sanitisation_window = 1.f;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "sanitisation_window";
    prop.desc = "The maximum time differential on which to base skeleton"
                "sanitisation, in seconds.";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->sanitisation_window;
    prop.float_state.min = 0.f;
    prop.float_state.max = 10.f;
    ctx->properties.push_back(prop);

    ctx->joint_velocity_sanitisation = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "joint_velocity_sanitisation";
    prop.desc = "If any joint position exceeds set velocity thresholds, "
                "use the displacement from the last well-tracked frame.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->joint_velocity_sanitisation;
    ctx->properties.push_back(prop);

    ctx->bone_length_sanitisation = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bone_length_sanitisation";
    prop.desc = "If bone length exceeds a set threshold of difference "
                "between previous tracked frames, use a previous length.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->bone_length_sanitisation;
    ctx->properties.push_back(prop);

    ctx->bone_rotation_sanitisation = false;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "bone_rotation_sanitisation";
    prop.desc = "If bone rotation exceeds a set threshold of difference "
                "between previous tracked frames, use a previous length.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->bone_rotation_sanitisation;
    ctx->properties.push_back(prop);

    ctx->use_bone_map_annotation = true;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "use_bone_map_annotation";
    prop.desc = "Use bone map annotations during sanitisation to determine "
                "if bones are likely to be realistic inferences.";
    prop.type = GM_PROPERTY_BOOL;
    prop.bool_state.ptr = &ctx->use_bone_map_annotation;
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

    ctx->skeleton_max_distance = 0.15f;
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

    if (ctx->label_map) {
        JSON_Array* label_map_array = json_array(ctx->label_map);

        for (int i = 0; i < ctx->n_labels; i++) {
            JSON_Object *mapping = json_array_get_object(label_map_array, i);

            enumerant = gm_ui_enumerant();
            enumerant.name = strdup(json_object_get_string(mapping, "name"));
            enumerant.desc = strdup(enumerant.name);
            enumerant.val = i;
            ctx->label_enumerants.push_back(enumerant);
        }
    } else {
        for (int i = 0; i < ctx->n_labels; i++) {
            char tmp_name[256];
            xsnprintf(tmp_name, sizeof(tmp_name), "Label %d", i);
            enumerant = gm_ui_enumerant();
            enumerant.name = strdup(tmp_name);
            enumerant.desc = strdup(enumerant.name);
            enumerant.val = i;
            ctx->label_enumerants.push_back(enumerant);
        }
    }

    prop.enum_state.n_enumerants = ctx->label_enumerants.size();
    prop.enum_state.enumerants = ctx->label_enumerants.data();
    ctx->properties.push_back(prop);

    ctx->debug_cloud_mode = DEBUG_CLOUD_MODE_VIDEO;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "cloud_mode";
    prop.desc = "debug mode for point cloud visualization";
    prop.type = GM_PROPERTY_ENUM;
    prop.enum_state.ptr = &ctx->debug_cloud_mode;

    enumerant = gm_ui_enumerant();
    enumerant.name = "none";
    enumerant.desc = "Don't create a debug point cloud";
    enumerant.val = DEBUG_CLOUD_MODE_NONE;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "video";
    enumerant.desc = "Texture with video";
    enumerant.val = DEBUG_CLOUD_MODE_VIDEO;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "depth";
    enumerant.desc = "Texture with depth";
    enumerant.val = DEBUG_CLOUD_MODE_DEPTH;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "class";
    enumerant.desc = "Motion classification";
    enumerant.val = DEBUG_CLOUD_MODE_CODEBOOK_LABELS;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "labels";
    enumerant.desc = "Body part labels";
    enumerant.val = DEBUG_CLOUD_MODE_LABELS;
    ctx->cloud_mode_enumerants.push_back(enumerant);

    enumerant = gm_ui_enumerant();
    enumerant.name = "edges";
    enumerant.desc = "Edges";
    enumerant.val = DEBUG_CLOUD_MODE_EDGES;
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
                                   tracking_create_rgb_video,
                               });

        stage.images.push_back((struct image_generator)
                               {
                                   "depth",
                                   "Depth frame acquired from camera",
                                   tracking_create_rgb_depth,
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
        enum tracking_stage stage_id = TRACKING_STAGE_NEAR_FAR_CULL_AND_INFILL;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "near_far_cull_and_infill";
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

        ctx->clamp_to_max_depth = true;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "clamp_to_max_depth";
        prop.desc = "Clamp the maximum depth of points instead of culling far points";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->clamp_to_max_depth;
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
        enum tracking_stage stage_id = TRACKING_STAGE_EDGE_DETECT;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "edge_detect";
        stage.desc = "Detect gradients coincident with viewing angle";

        ctx->edge_detect_mode = EDGE_DETECT_MODE_XY;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "edge_detect_mode";
        prop.desc = "Edge detection strategy";
        prop.type = GM_PROPERTY_ENUM;
        prop.enum_state.ptr = &ctx->edge_detect_mode;

        enumerant = gm_ui_enumerant();
        enumerant.name = "none";
        enumerant.desc = "Don't detect any edges";
        enumerant.val = EDGE_DETECT_MODE_NONE;
        ctx->edge_detect_mode_enumerants.push_back(enumerant);

        enumerant = gm_ui_enumerant();
        enumerant.name = "x_only";
        enumerant.desc = "Only detect horizontal edges";
        enumerant.val = EDGE_DETECT_MODE_X_ONLY;
        ctx->edge_detect_mode_enumerants.push_back(enumerant);

        enumerant = gm_ui_enumerant();
        enumerant.name = "y_only";
        enumerant.desc = "Only detect vertical edges";
        enumerant.val = EDGE_DETECT_MODE_Y_ONLY;
        ctx->edge_detect_mode_enumerants.push_back(enumerant);

        enumerant = gm_ui_enumerant();
        enumerant.name = "xy";
        enumerant.desc = "Detect horizontal and vertical edges";
        enumerant.val = EDGE_DETECT_MODE_XY;
        ctx->edge_detect_mode_enumerants.push_back(enumerant);

        prop.enum_state.n_enumerants = ctx->edge_detect_mode_enumerants.size();
        prop.enum_state.enumerants = ctx->edge_detect_mode_enumerants.data();
        stage.properties.push_back(prop);

        ctx->edge_threshold = 0.99f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "edge_detect_threshold";
        prop.desc = "Threshold for considering eye vector and gradient coincident";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->edge_threshold;
        prop.float_state.min = 0;
        prop.float_state.max = 1;
        stage.properties.push_back(prop);

        ctx->edge_break_x = -1;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "edge_detect_break_x";
        prop.desc = "Focus debugging on a specific point during edge detect";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->edge_break_x;
        prop.int_state.min = -1;
        prop.int_state.max = 320; // FIXME: set according to resolution
        stage.properties.push_back(prop);

        ctx->edge_break_y = -1;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "edge_detect_break_y";
        prop.desc = "Focus debugging on a specific point during edge detect";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->edge_break_y;
        prop.int_state.min = -1;
        prop.int_state.max = 240; // FIXME: set according to resolution
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
        enum tracking_stage stage_id = TRACKING_STAGE_NAIVE_FLOOR;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "naive_find_floor";
        stage.desc = "Find floor to attempt naive single-person segmentation";

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

        ctx->cluster_from_prev = true;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cluster_from_prev";
        prop.desc = "During naive segmentation, cluster from the positions of "
                    "tracked joints on previous frames";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->cluster_from_prev;
        stage.properties.push_back(prop);

        ctx->cluster_from_prev_dist_threshold = 0.1f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cluster_from_prev_dist_threshold";
        prop.desc = "The maximum distance between the point in an old frame "
                    "and new before not considering it for clustering.";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_from_prev_dist_threshold;
        prop.float_state.min = 0.01f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->cluster_from_prev_time_threshold = 0.5f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cluster_from_prev_time_threshold";
        prop.desc = "The maximum time difference when using a previous frame "
                    "to determine clustering start points, in seconds.";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->cluster_from_prev_time_threshold;
        prop.float_state.min = 0.f;
        prop.float_state.max = 1.f;
        stage.properties.push_back(prop);

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

        ctx->cluster_tolerance = 0.05f;
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
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_RETIRE_WORDS;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_retire";
        stage.desc = "Retire old codewords";

        // XXX: aliased property
        ctx->codebook_frozen = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_frozen";
        prop.desc = "Disable any further codebook updates while true";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->codebook_frozen;
        stage.properties.push_back(prop);

        ctx->codeword_timeout = 3.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_timeout";
        prop.desc = "Codewords that don't match any points after this timeout will be removed";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codeword_timeout;
        prop.float_state.min = 0.2f;
        prop.float_state.max = 10.f;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->codebook_foreground_scrub_timeout = 1.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_foreground_scrub_timeout";
        prop.desc = "If we haven't tracked for this long (seconds) then try deleting all but the farthest codewords";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_foreground_scrub_timeout;
        prop.float_state.min = 0.0f;
        prop.float_state.max = 60.0f;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->codebook_clear_timeout = 3.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_clear_timeout";
        prop.desc = "If we haven't tracked for this long (seconds) then try clearing the full codebook";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_clear_timeout;
        prop.float_state.min = 0.0f;
        prop.float_state.max = 60.0f;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->codebook_keep_back_most_threshold = 0.2f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_keep_back_most_threshold";
        prop.desc = "When tracking fails we throw away all but the farthest codewords, within a threshold band this deep";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_keep_back_most_threshold;
        prop.float_state.min = 0.0f;
        prop.float_state.max = 1.0f;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->debug_codebook_layer = 0;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "debug_codebook_layer";
        prop.desc = "If != 0 then only show the Nth (base 1) codeword for each codebook entry (positive counts from nearest, negative counts from farthest)";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->debug_codebook_layer;
        prop.int_state.min = -15;
        prop.int_state.max = 15;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->codebook_debug_view = CODEBOOK_DEBUG_VIEW_POINT_CLOUD;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_debug_view";
        prop.desc = "Whether to visualize the codebook or the point cloud being processed";
        prop.type = GM_PROPERTY_ENUM;
        prop.enum_state.ptr = &ctx->codebook_debug_view;

        enumerant = gm_ui_enumerant();
        enumerant.name = "point_cloud";
        enumerant.desc = "Visualize the point cloud being processed";
        enumerant.val = CODEBOOK_DEBUG_VIEW_POINT_CLOUD;
        ctx->codebook_debug_view_enumerants.push_back(enumerant);

        enumerant = gm_ui_enumerant();
        enumerant.name = "codebook";
        enumerant.desc = "Visualize the codebook";
        enumerant.val = CODEBOOK_DEBUG_VIEW_CODEBOOK;
        ctx->codebook_debug_view_enumerants.push_back(enumerant);

        prop.enum_state.n_enumerants = ctx->codebook_debug_view_enumerants.size();
        prop.enum_state.enumerants = ctx->codebook_debug_view_enumerants.data();
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_RESOLVE_BACKGROUND;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_resolve_bg";
        stage.desc = "Determine canonical background codewords plus prune old codewords";

        // XXX: aliased property
        ctx->debug_codebook_layer = 0;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "debug_codebook_layer";
        prop.desc = "If != 0 then only show the Nth (base 1) codeword for each codebook entry (positive counts from nearest, negative counts from farthest)";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->debug_codebook_layer;
        prop.int_state.min = -15;
        prop.int_state.max = 15;
        stage.properties.push_back(prop);

        // XXX: aliased property
        ctx->codebook_debug_view = CODEBOOK_DEBUG_VIEW_POINT_CLOUD;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_debug_view";
        prop.desc = "Whether to visualize the codebook or the point cloud being processed";
        prop.type = GM_PROPERTY_ENUM;
        prop.enum_state.ptr = &ctx->codebook_debug_view;
        prop.enum_state.n_enumerants = ctx->codebook_debug_view_enumerants.size();
        prop.enum_state.enumerants = ctx->codebook_debug_view_enumerants.data();
        stage.properties.push_back(prop);

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
        enum tracking_stage stage_id = TRACKING_STAGE_CODEBOOK_CLASSIFY;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "codebook_classify";
        stage.desc = "Analyse and classify motion in codebook space for segmentation";
        stage.images.push_back((struct image_generator)
                               {
                                   "codebook_classifications",
                                   "Codebook classifications from motion analysis",
                                   tracking_create_rgb_depth_classification,
                               });

        ctx->codebook_bg_threshold = 0.05f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_bg_threshold";
        prop.desc = "Threshold distance from back-most depth values to be classed as 'background'";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_bg_threshold;
        prop.float_state.min = 0.001f;
        prop.float_state.max = 0.3f;
        stage.properties.push_back(prop);

        ctx->codebook_flat_threshold = 0.2f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_flat_threshold";
        prop.desc = "Threshold distance from back-most depth values to classes as 'flat' if not 'background'";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_flat_threshold;
        prop.float_state.min = 0.05f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->codeword_mean_n_max = 100;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_mean_n_max";
        prop.desc = "When updating rolling average depth of codeword, n is clamped to max so new values don't loose all influence";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codeword_mean_n_max;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->codeword_flicker_max_run_len = 3;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_flicker_max_run_len";
        prop.desc = "The maximum number of consecutive updates a codeword can see (averaged out) to still be considered as 'flickering'";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codeword_flicker_max_run_len;
        prop.int_state.min = 1;
        prop.int_state.max = 10;
        stage.properties.push_back(prop);

        ctx->codeword_flicker_max_quiet_frames = 100;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_flicker_max_quiet_frames";
        prop.desc = "The maximum number of frames that can elapse before a (short) update sequence is seen for a codeword to possibly be considered 'flickering'";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codeword_flicker_max_quiet_frames;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->codeword_obj_min_n = 200;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_obj_min_n";
        prop.desc = "The minimum number of matches/updates before a codeword might be classed as an (ignored) foreground object";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codeword_obj_min_n;
        prop.int_state.min = 10;
        prop.int_state.max = 500;
        stage.properties.push_back(prop);

        ctx->codeword_obj_max_frame_to_n_ratio = 0.8f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codeword_obj_max_frame_to_n_ratio";
        prop.desc = "The maximum ratio of frames per n codeword updates for a codeword to possibly be classed as an (ignored) foreground object";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codeword_obj_max_frame_to_n_ratio;
        prop.float_state.min = 0.f;
        prop.float_state.max = 1.f;
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
                                   tracking_create_rgb_candidate_clusters,
                               });

        ctx->codebook_cluster_tolerance = 0.10f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "cluster_tolerance";
        prop.desc = "Distance threshold when clustering classified points";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_cluster_tolerance;
        prop.float_state.min = 0.01f;
        prop.float_state.max = 0.5f;
        stage.properties.push_back(prop);

        ctx->codebook_tiny_cluster_threshold = 10;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "tiny_cluster_threshold";
        prop.desc = "Clusters with fewer points will be considered noise and possibly merged into larger candidate clusters";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codebook_tiny_cluster_threshold;
        prop.int_state.min = 0;
        prop.int_state.max = 100;
        stage.properties.push_back(prop);

        ctx->codebook_large_cluster_threshold = 200;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "large_cluster_threshold";
        prop.desc = "Clusters with at least this many points may be merged with 'tiny' clusters (based on cluster_tolerance distance thresholding)";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->codebook_large_cluster_threshold;
        prop.int_state.min = 0;
        prop.int_state.max = 2000;
        stage.properties.push_back(prop);

        ctx->codebook_cluster_infill = true;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_cluster_infill";
        prop.desc = "Merge touching tiny clusters into large clusters";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->codebook_cluster_infill;
        stage.properties.push_back(prop);

        ctx->codebook_cluster_merge_large_neighbours = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_cluster_merge_large_neighbours";
        prop.desc = "Merge large, adjacent clusters";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->codebook_cluster_merge_large_neighbours;
        stage.properties.push_back(prop);

        ctx->debug_codebook_cluster_idx = -1;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "debug_codebook_cluster_idx";
        prop.desc = "Only view this (large) cluster (show all if == -1)";
        prop.type = GM_PROPERTY_INT;
        prop.int_state.ptr = &ctx->debug_codebook_cluster_idx;
        prop.int_state.min = -1;
        prop.int_state.max = 20;
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
        enum tracking_stage stage_id = TRACKING_STAGE_CROP_CLUSTER_IMAGE;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "crop_cluster_image";
        stage.desc = "Create a cropped 2D depth buffer from a candidate cluster";

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_SELECT_CANDIDATE_CLUSTER;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "select_cluster";
        stage.desc = "Select cluster to run label inference on (points before projection into depth image)";

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
                                   tracking_create_rgb_label_map,
                               });

        ctx->use_threads = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "li_use_threads";
        prop.desc = "Use extra threads during label inference";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->use_threads;
        stage.properties.push_back(prop);

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

        ctx->fast_clustering = true;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "li_fast_clustering";
        prop.desc = "Use 2D connected-points clustering during joint inference";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->fast_clustering;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_REFINE_SKELETON;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "refine_skeleton";
        stage.desc = "Try to verify the best inferred skeleton joints "
                     "have been chosen";

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
        stage.desc = "Try and clean up issues with the derived skeleton";

        ctx->joint_velocity_threshold = 1.5f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "joint_velocity_threshold";
        prop.desc = "Minimum travel velocity (m/s) "
                    "before considering a joint for error correction";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->joint_velocity_threshold;
        prop.float_state.min = 0.3f;
        prop.float_state.max = 3.0f;
        stage.properties.push_back(prop);

        ctx->joint_outlier_factor = 2.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "joint_outlier_factor";
        prop.desc = "The factor by which a joint velocity can deviate "
                    "from the average velocity before being sanitised";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->joint_outlier_factor;
        prop.float_state.min = 1.0f;
        prop.float_state.max = 5.0f;
        stage.properties.push_back(prop);

        ctx->bone_length_outlier_factor = 1.1f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "bone_length_outlier_factor";
        prop.desc = "The factor by which a bone length can deviate from the "
                    "average length before being sanitised";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->bone_length_outlier_factor;
        prop.float_state.min = 1.0f;
        prop.float_state.max = 5.0f;
        stage.properties.push_back(prop);

        ctx->bone_rotation_outlier_factor = 2.0f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "bone_rotation_outlier_factor";
        prop.desc = "The factor by which a bone's rotation can deviate from "
                    "the average rotation variation before being sanitised";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->bone_rotation_outlier_factor;
        prop.float_state.min = 1.0f;
        prop.float_state.max = 5.0f;
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

        ctx->codebook_clear_tracked_threshold = 0.15f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_clear_tracked_threshold";
        prop.desc = "The depth distance threshold (meters) to use when clearing codewords corresponding to tracked points";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->codebook_clear_tracked_threshold;
        prop.float_state.min = 0.0f;
        prop.float_state.max = 1.0f;
        stage.properties.push_back(prop);

        // XXX: this an alias of a property set up earlier...
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "codebook_debug_view";
        prop.desc = "Whether to visualize the codebook or the point cloud being processed";
        prop.type = GM_PROPERTY_ENUM;
        prop.enum_state.ptr = &ctx->codebook_debug_view;

        prop.enum_state.n_enumerants = ctx->codebook_debug_view_enumerants.size();
        prop.enum_state.enumerants = ctx->codebook_debug_view_enumerants.data();
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    {
        enum tracking_stage stage_id = TRACKING_STAGE_UPDATE_HISTORY;
        struct gm_pipeline_stage &stage = ctx->stages[stage_id];

        stage.stage_id = stage_id;
        stage.name = "update_history";
        stage.desc = "Update the tracking history with new results";

        ctx->debug_predictions = false;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "debug_predictions";
        prop.desc = "Whether to visualize how a skeleton prediction is derived from the latest tracking history";
        prop.type = GM_PROPERTY_BOOL;
        prop.bool_state.ptr = &ctx->debug_predictions;
        stage.properties.push_back(prop);

        ctx->debug_prediction_offset = -0.1f;
        prop = gm_ui_property();
        prop.object = ctx;
        prop.name = "debug_prediction_delay";
        prop.desc = "The offset from the most recent tracking timestamp to use when requesting a prediction to visualize";
        prop.type = GM_PROPERTY_FLOAT;
        prop.float_state.ptr = &ctx->debug_prediction_offset;
        prop.float_state.min = -2.0f;
        prop.float_state.max = 1.0f;
        stage.properties.push_back(prop);

        stage.properties_state.n_properties = stage.properties.size();
        stage.properties_state.properties = stage.properties.data();
        pthread_mutex_init(&stage.properties_state.lock, NULL);
    }

    for (int i = 1; i < N_TRACKING_STAGES; i++) {
        gm_assert(ctx->log, ctx->stages[i].stage_id != TRACKING_STAGE_START,
                  "Uninitialized stage description %d", i);
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

uint64_t
gm_context_get_stage_frame_duration_avg(struct gm_context *ctx,
                                        int stage_index)
{
    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage &stage = ctx->stages[stage_index];

    uint64_t ret;

    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);
    if (stage.n_frames)
        ret = stage.total_time_ns / stage.n_frames;
    else
        ret = 0;
    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);

    return ret;
}

uint64_t
gm_context_get_stage_frame_duration_median(struct gm_context *ctx,
                                           int stage_index)
{
    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage &stage = ctx->stages[stage_index];

    if (stage.frame_duration_hist.size() <= 1)
        return stage.total_time_ns;

    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);
    int len = stage.frame_duration_hist.size();
    uint32_t tmp[len]; // Assume durations less than 4.3 seconds
    for (int i = 0; i < len; i++)
        tmp[i] = (uint32_t)stage.frame_duration_hist[i];
    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);

    std::sort(tmp, tmp + len);

    return tmp[len/2];
}

uint64_t
gm_context_get_stage_run_duration_avg(struct gm_context *ctx,
                                      int stage_index)
{
    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage &stage = ctx->stages[stage_index];

    uint64_t ret;

    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);
    if (stage.n_frames)
        ret = stage.total_time_ns / stage.n_invocations;
    else
        ret = 0;
    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);

    return ret;
}

uint64_t
gm_context_get_stage_run_duration_median(struct gm_context *ctx,
                                         int stage_index)
{
    gm_assert(ctx->log, stage_index >=0 && stage_index < (int)ctx->stages.size(),
              "Out of range stage index");

    struct gm_pipeline_stage &stage = ctx->stages[stage_index];

    if (stage.invocation_duration_hist.size() <= 1)
        return stage.total_time_ns;

    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);
    int len = stage.invocation_duration_hist.size();
    uint32_t tmp[len]; // Assume durations less than 4.3 seconds
    for (int i = 0; i < len; i++)
        tmp[i] = (uint32_t)stage.invocation_duration_hist[i];
    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);

    std::sort(tmp, tmp + len);

    return tmp[len/2];
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
gm_context_get_average_frame_duration(struct gm_context *ctx)
{
    uint64_t ret;

    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);
    if (ctx->n_frames <= 1)
        ret = ctx->total_tracking_duration;
    else
        ret = ctx->total_tracking_duration / ctx->n_frames;
    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);

    return ret;
}

int
gm_context_get_n_joints(struct gm_context *ctx)
{
    gm_assert(ctx->log, ctx->joint_names.size() != 0, "No joint map");

    return ctx->joint_names.size();
}

const char *
gm_context_get_joint_name(struct gm_context *ctx, int joint_id)
{
    gm_assert(ctx->log, joint_id >= 0 && joint_id < ctx->joint_names.size(),
              "Invalid joint index");

    return ctx->joint_names[joint_id];
}

const enum gm_joint_semantic
gm_context_get_joint_semantic(struct gm_context *ctx, int joint_id)
{
    gm_assert(ctx->log, joint_id >= 0 && joint_id < ctx->joint_names.size(),
              "Invalid joint index");

    return ctx->joint_semantics[joint_id];
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

static void
context_clear_metrics(struct gm_context *ctx)
{
    pthread_mutex_lock(&ctx->aggregate_metrics_mutex);

    /* Clear all metrics */
    ctx->total_tracking_duration = 0;
    ctx->n_frames = 0;

    for (int i = 0; i < ctx->stages.size(); i++) {

        struct gm_pipeline_stage &stage = ctx->stages[i];

        stage.frame_duration_hist.clear();
        stage.frame_duration_hist_head = 0;
        stage.invocation_duration_hist.clear();
        stage.invocation_duration_hist_head = 0;
    }

    pthread_mutex_unlock(&ctx->aggregate_metrics_mutex);
}

void
gm_context_flush(struct gm_context *ctx, char **err)
{
    stop_tracking_thread(ctx);

    /* XXX: we don't need to hold the tracking_swap_mutex here because we've
     * stopped the tracking thread...
     */
    context_clear_tracking_locked(ctx,
                                  true); // and clear tracking/prediction pools

    context_clear_metrics(ctx);

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    if (ctx->frame_ready) {
        gm_frame_unref(ctx->frame_ready);
        ctx->frame_ready = NULL;
    }
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    ctx->stopping = false;
    gm_debug(ctx->log, "Glimpse context flushed, restarting tracking thread");

    int ret = start_tracking_thread(ctx, NULL);
    if (ret != 0) {
        gm_throw(ctx->log, err,
                 "Failed to start tracking thread: %s", strerror(ret));
    }
}

struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx)
{
    struct gm_tracking *tracking = NULL;

    pthread_mutex_lock(&ctx->tracking_swap_mutex);
    if (ctx->latest_paused_tracking) {
        tracking = gm_tracking_ref(&ctx->latest_paused_tracking->base);

        gm_debug(ctx->log, "get_latest_tracking = %p (ref = %d) (paused)\n",
                 ctx->latest_paused_tracking,
                 atomic_load(&ctx->latest_paused_tracking->base.ref));
    } else if (ctx->latest_tracking) {
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
        prediction->tracking_history[closest_frame]->skeleton_corrected;
    prediction->skeleton = closest_skeleton;

    if (ctx->prediction_dampen_large_deltas) {
        timestamp = calculate_decayed_timestamp(
            prediction->tracking_history[closest_frame]->frame->timestamp,
            timestamp, ctx->max_prediction_delta, ctx->prediction_decay);
    }
    prediction->timestamp = timestamp;

    uint64_t closest_timestamp =
        prediction->tracking_history[closest_frame]->frame->timestamp;
    if (timestamp == closest_timestamp || prediction->n_tracking <= 1) {
        prediction->h1 = closest_frame;
        prediction->h2 = -1;
        return &prediction->base;
    }

    // Work out the two nearest frames and the interpolation value
    int h1;
    if (timestamp > closest_timestamp) {
        h1 = (closest_frame == 0) ? 0 : closest_frame - 1;
    } else {
        h1 = (closest_frame == prediction->n_tracking - 1) ?
            closest_frame - 1 : closest_frame;
    }
    int h2 = h1 + 1;

    prediction->h1 = h1;
    prediction->h2 = h2;

    struct gm_tracking_impl *frame1 = prediction->tracking_history[h1];
    struct gm_tracking_impl *frame2 = prediction->tracking_history[h2];
    float t = (timestamp - frame2->frame->timestamp) /
              (float)(frame1->frame->timestamp - frame2->frame->timestamp);

    int n_bones = ctx->n_bones;

    for (int b = 0; b < n_bones; b++) {
        struct gm_bone &bone = closest_skeleton.bones[b];
        struct gm_bone_info &bone_info = ctx->bone_info[b];

        if (!frame2->skeleton_corrected.bones[b].valid ||
            !frame1->skeleton_corrected.bones[b].valid) {
            continue;
        }

        // As a special case; use linear interpolation to place the root bone
        if (bone_info.parent < 0 ||
            ctx->prediction_interpolate_angles == false)
        {
            interpolate_joints(
                frame2->skeleton_corrected.joints[bone_info.head],
                frame1->skeleton_corrected.joints[bone_info.head],
                t, prediction->skeleton.joints[bone_info.head]);
            interpolate_joints(
                frame2->skeleton_corrected.joints[bone_info.tail],
                frame1->skeleton_corrected.joints[bone_info.tail],
                t, prediction->skeleton.joints[bone_info.tail]);
            continue;
        }

        struct gm_bone &parent_bone = closest_skeleton.bones[bone_info.parent];
        struct gm_bone_info &parent_bone_info = ctx->bone_info[parent_bone.idx];

        struct gm_bone &frame1_bone = frame1->skeleton_corrected.bones[bone.idx];
        struct gm_bone &frame2_bone = frame2->skeleton_corrected.bones[bone.idx];

        // Find the angle to rotate the parent bone. Note, we're relying
        // on bones being stored in an order where we can rely on the
        // bone's parent being seen before any descendants.
        glm::mat3 rotate = glm::mat3_cast(
            glm::slerp(frame2_bone.angle, frame1_bone.angle, t));

        struct gm_joint &parent_head =
            prediction->skeleton.joints[parent_bone_info.head];
        struct gm_joint &parent_tail =
            prediction->skeleton.joints[parent_bone_info.tail];

        glm::vec3 parent_vec = glm::normalize(
            glm::vec3(parent_tail.x - parent_head.x,
                      parent_tail.y - parent_head.y,
                      parent_tail.z - parent_head.z));
        glm::vec3 new_tail = ((parent_vec * rotate) * bone.length);
        new_tail.x += prediction->skeleton.joints[bone_info.head].x;
        new_tail.y += prediction->skeleton.joints[bone_info.head].y;
        new_tail.z += prediction->skeleton.joints[bone_info.head].z;

        prediction->skeleton.joints[bone_info.tail].x = new_tail.x;
        prediction->skeleton.joints[bone_info.tail].y = new_tail.y;
        prediction->skeleton.joints[bone_info.tail].z = new_tail.z;
    }

    update_bones(ctx, prediction->skeleton);

    return &prediction->base;
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


    gm_info(ctx->log, "Downsampling via GLES");

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
        gm_info(ctx->log, "Orientation change to account for with face detection");
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
            gm_info(ctx->log, "> rotation = 0");
            break;
        case GM_ROTATION_90:
            need_portrait_downsample_fb = false;
            gm_info(ctx->log, "> rotation = 90");
            break;
        case GM_ROTATION_180:
            need_portrait_downsample_fb = true;
            gm_info(ctx->log, "> rotation = 180");
            break;
        case GM_ROTATION_270:
            need_portrait_downsample_fb = false;
            gm_info(ctx->log, "> rotation = 270");
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
    gm_info(ctx->log, "rotated frame width = %d, height = %d",
            (int)rotated_frame_width, (int)rotated_frame_height);

    if (need_portrait_downsample_fb != have_portrait_downsample_fb_) {
        if (downsample_fbo_) {
            gm_info(ctx->log, "Discarding previous downsample fbo and texture");
            glDeleteFramebuffers(1, &downsample_fbo_);
            downsample_fbo_ = 0;
            glDeleteTextures(1, &downsample_tex2d_);
            downsample_tex2d_ = 0;
        }
        if (ctx->read_back_fbo) {
            gm_info(ctx->log, "Discarding previous read_back_fbo and texture");
            glDeleteFramebuffers(1, &ctx->read_back_fbo);
            ctx->read_back_fbo = 0;
        }
    }

    if (!downsample_fbo_) {
        gm_info(ctx->log, "Allocating new %dx%d downsample fbo + texture",
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
            gm_error(ctx->log, "Framebuffer complete check (for downsample fbo) failed");

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

        gm_info(ctx->log, "Created level0 scale shader");
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

        gm_info(ctx->log, "Created scale shader");
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

    gm_info(ctx->log,
            "Uploaded top level luminance texture to GPU via glTexSubImage2D in %.3f%s",
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

    //gm_info(ctx->log, "Allocated pyramid level texture + fbo in %.3f%s",
    //        get_duration_ns_print_scale(duration_ns),
    //        get_duration_ns_print_scale_suffix(duration_ns));

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

    gm_info(ctx->log, "Submitted level0 downsample in %.3f%s",
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

    gm_info(ctx->log, "glGenerateMipmap took %.3f%s",
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
            gm_error(ctx->log, "Famebuffer complete check failed");

        //gm_info(ctx->log, "Allocated pyramid level texture + fbo in %.3f%s",
        //        get_duration_ns_print_scale(duration_ns),
        //        get_duration_ns_print_scale_suffix(duration_ns));

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

    gm_info(ctx->log, "glReadPixels took %.3f%s",
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

    gm_info(ctx->log, "glMapBufferRange took %.3f%s",
            get_duration_ns_print_scale(duration_ns),
            get_duration_ns_print_scale_suffix(duration_ns));

    {
        dlib::timing::timer lv0_cpy_timer("Copied pyramid level0 frame for face detection from PBO in");

        tracking->face_detect_buf_width = rotated_frame_width / 4;
        tracking->face_detect_buf_height = rotated_frame_height / 4;

        /* TODO: avoid copying out of the PBO later (assuming we can get a
         * cached mapping)
         */
        gm_info(ctx->log, "face detect scratch width = %d, height = %d",
                (int)tracking->face_detect_buf_width,
                (int)tracking->face_detect_buf_height);
        ctx->grey_face_detect_scratch.resize(tracking->face_detect_buf_width *
                                             tracking->face_detect_buf_height);
        memcpy(ctx->grey_face_detect_scratch.data(),
               pbo_ptr, ctx->grey_face_detect_scratch.size());

        tracking->face_detect_buf = ctx->grey_face_detect_scratch.data();
        gm_info(ctx->log, "tracking->face_detect_buf = %p",
                tracking->face_detect_buf);
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

int
gm_bone_get_head(struct gm_context *ctx,
                 const struct gm_bone *bone)
{
    return ctx->bone_info[bone->idx].head;
}

int
gm_bone_get_tail(struct gm_context *ctx,
                 const struct gm_bone *bone)
{
    return ctx->bone_info[bone->idx].tail;
}

float
gm_bone_get_length(struct gm_context *ctx,
                   const struct gm_bone *bone)
{
    return bone->length;
}

void
gm_bone_get_angle(struct gm_context *ctx,
                  const struct gm_bone *bone,
                  float *out_xyzw)
{
    if (out_xyzw) {
        out_xyzw[0] = bone->angle.x;
        out_xyzw[1] = bone->angle.y;
        out_xyzw[2] = bone->angle.z;
        out_xyzw[3] = bone->angle.w;
    }
}

struct gm_skeleton *
gm_skeleton_new(struct gm_context *ctx, struct gm_joint *joints)
{
    struct gm_skeleton *skeleton = new gm_skeleton();

    skeleton->ctx = ctx;

    skeleton->joints.resize(ctx->n_joints);
    for (int j = 0; j < ctx->n_joints; ++j) {
        skeleton->joints[j] = joints[j];
    }

    update_bones(ctx, *skeleton);

    return skeleton;
}

static void
resize_bone_and_children(struct gm_context *ctx,
                         struct gm_skeleton *original_skeleton,
                         struct gm_skeleton *ref_skeleton,
                         struct gm_skeleton *resized_skeleton,
                         int bone_idx)
{
    struct gm_bone_info &bone_info = ctx->bone_info[bone_idx];

    float length_scale = ref_skeleton->bones[bone_idx].length /
        original_skeleton->bones[bone_idx].length;

    resized_skeleton->bones[bone_idx].length = ref_skeleton->bones[bone_idx].length;

    struct gm_joint &orig_head = original_skeleton->joints[bone_info.head];
    struct gm_joint &orig_tail = original_skeleton->joints[bone_info.tail];

    // NB: when we modify this tail joint we will also be moving the
    // head of child bones.
    //
    // We are careful to ignore the relative positions of joints in
    // the skeleton being resized due to this temporarily inconsistent
    // state.
    struct gm_joint &head = resized_skeleton->joints[bone_info.head];
    struct gm_joint &tail = resized_skeleton->joints[bone_info.tail];
    tail.x = head.x + (orig_tail.x - orig_head.x) * length_scale;
    tail.y = head.y + (orig_tail.y - orig_head.y) * length_scale;
    tail.z = head.z + (orig_tail.z - orig_head.z) * length_scale;

    for (int i = 0; i < bone_info.n_children; i++) {
        resize_bone_and_children(ctx,
                                 original_skeleton,
                                 ref_skeleton,
                                 resized_skeleton,
                                 bone_info.children[i]);
    }
}

struct gm_skeleton *
gm_skeleton_resize(struct gm_context *ctx,
                   struct gm_skeleton *skeleton,
                   struct gm_skeleton *ref_skeleton,
                   int anchor_joint_idx)
{
    if (skeleton->bones.size() != ref_skeleton->bones.size()) {
        gm_error(ctx->log,
                 "Mismatching skeletons passed to gm_skeleton_resize(): skel n_joints=%d, n_bones=%d, ref n_joints=%d, n_bones=%d",
                 (int)skeleton->joints.size(),
                 (int)skeleton->bones.size(),
                 (int)ref_skeleton->joints.size(),
                 (int)ref_skeleton->bones.size());
        return NULL;
    }

    struct gm_skeleton *resized = new gm_skeleton();

    resized->ctx = ctx;
    resized->joints = skeleton->joints;
    resized->bones = skeleton->bones;

    gm_assert(ctx->log, ctx->bone_info[0].parent == -1,
              "Expected the first bone to be the root bone");

    resize_bone_and_children(ctx,
                             skeleton,
                             ref_skeleton,
                             resized,
                             0); // root bone index

    // The anchor joint is the one joint that shouldn't appear to move after
    // the resize so now just translate the resized skeleton so it's anchor
    // joint matches the ref_skeleton's corresponding anchor joint

    struct gm_joint &ref_anchor_joint = ref_skeleton->joints[anchor_joint_idx];
    struct gm_joint &resized_anchor_joint = resized->joints[anchor_joint_idx];
    float anchor_delta[3] = {
        ref_anchor_joint.x - resized_anchor_joint.x,
        ref_anchor_joint.y - resized_anchor_joint.y,
        ref_anchor_joint.z - resized_anchor_joint.z,
    };

    int n_joints = ctx->n_joints;
    for (int i = 0; i < n_joints; i++) {
        struct gm_joint &joint = resized->joints[i];

        joint.x += anchor_delta[0];
        joint.y += anchor_delta[1];
        joint.z += anchor_delta[2];
    }

    update_bones(ctx, *resized);

    return resized;
}

bool
gm_skeleton_save(const struct gm_skeleton *skeleton,
                 const char *filename)
{
    int n_joints = gm_skeleton_get_n_joints(skeleton);

    for (int i = 0; i < n_joints; i++) {
        const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);
        if (!joint || !joint->valid)
            return false;
    }

    JSON_Value *root = json_value_init_object();
    JSON_Value *joints = json_value_init_array();
    json_object_set_value(json_object(root), "joints", joints);

    for (int i = 0; i < n_joints; i++) {
        const struct gm_joint *joint = gm_skeleton_get_joint(skeleton, i);
        JSON_Value *joint_js = json_value_init_object();
        json_object_set_number(json_object(joint_js), "x", joint->x);
        json_object_set_number(json_object(joint_js), "y", joint->y);
        json_object_set_number(json_object(joint_js), "z", joint->z);
        json_array_append_value(json_array(joints), joint_js);
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
    if (skeleton->bones[bone].valid)
        return &skeleton->bones[bone];
    else
        return NULL;
}

const struct gm_joint *
gm_skeleton_get_joint(const struct gm_skeleton *skeleton, int joint)
{
    /* For now a NULL name is what indicates that we failed to track
     * this joint...
     */
    if (skeleton->joints[joint].valid)
        return &skeleton->joints[joint];
    else
        return NULL;
}
