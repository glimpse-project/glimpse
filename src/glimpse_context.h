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

#pragma once

#include <assert.h>

#ifdef __cplusplus
#include <atomic>
#define _Atomic(T) std::atomic<T>
#else
#include <stdatomic.h>
#endif

#include "glimpse_properties.h"
#include "glimpse_log.h"
#include "rdt_tree.h"

/* XXX: Disturbing the order of this enum will break recordings */
enum gm_format {
    GM_FORMAT_UNKNOWN,
    GM_FORMAT_Z_U16_MM,
    GM_FORMAT_Z_F32_M,
    GM_FORMAT_Z_F16_M,
    GM_FORMAT_LUMINANCE_U8,
    GM_FORMAT_RGB_U8,
    GM_FORMAT_RGBX_U8,
    GM_FORMAT_RGBA_U8,

    GM_FORMAT_POINTS_XYZC_F32_M, // points; not an image

    GM_FORMAT_BGR_U8,
    GM_FORMAT_BGRX_U8,
    GM_FORMAT_BGRA_U8,
};

enum gm_distortion_model {
    GM_DISTORTION_NONE,

    /* The 'FOV model' described in:
     * > Frédéric Devernay, Olivier Faugeras. Straight lines have to be straight:
     * > automatic calibration and re-moval of distortion from scenes of
     * > structured enviroments. Machine Vision and Applications, Springer
     * > Verlag, 2001, 13 (1), pp.14-24. <10.1007/PL00013269>. <inria-00267247>
     *
     * (for fish-eye lenses)
     */
    GM_DISTORTION_FOV_MODEL,

    /* Brown's distortion model, with k1, k2 parameters */
    GM_DISTORTION_BROWN_K1_K2,
    /* Brown's distortion model, with k1, k2, k3 parameters */
    GM_DISTORTION_BROWN_K1_K2_K3,
    /* Brown's distortion model, with k1, k2, p1, p2, k3 parameters */
    GM_DISTORTION_BROWN_K1_K2_P1_P2_K3,
};

struct gm_intrinsics {
  uint32_t width;
  uint32_t height;

  double fx;
  double fy;
  double cx;
  double cy;

  enum gm_distortion_model distortion_model;

  /* XXX: maybe we should hide these coeficients since we can't represent
   * more complex models e.g. using a triangle mesh
   */
  double distortion[5];
};

struct gm_extrinsics {
  float rotation[9];    // Column-major 3x3 rotation matrix
  float translation[3]; // Translation vector, in meters
};



enum gm_event_type
{
    GM_EVENT_REQUEST_FRAME,
    GM_EVENT_TRACKING_READY
};

#define GM_REQUEST_FRAME_DEPTH  1ULL<<0
#define GM_REQUEST_FRAME_VIDEO  1ULL<<1

struct gm_event
{
    enum gm_event_type type;

    union {
        struct {
            uint64_t flags;
        } request_frame;
    };
};

enum gm_pose_type
{
    GM_POSE_INVALID,    // Pose is invalid
    GM_POSE_TO_START,   // Pose transforms to a ground-aligned 'starting' pose.
    GM_POSE_TO_GROUND   // Pose orients to align with the ground.
};

struct gm_pose {
    enum gm_pose_type type;
    float orientation[4]; // x, y, z, w
    float translation[3];
};

/* XXX: beware a PCL PointXYZRGBA made of 3 floats + a uint32 rgba member
 * doesn't have a size of 16 bytes, it has a size of 32 bytes and the
 * typedefs in PCL are a tangle of macros and templates. We define our own
 * type for our C api...
 */
struct gm_point_rgba {
    float x, y, z;
    uint32_t rgba;
};

struct gm_buffer;

/* A reference to a single data buffer
 *
 * Used to reference count buffers attached to frames where we want to abstract
 * away the life-cycle management of the underlying allocation/storage.
 *
 * Frames will be comprised of multiple buffers which themselves may be the
 * product of more than one device (e.g. depth + rgb cameras and accelerometer
 * data buffers) Each type of buffer might be associated with a different pool
 * or swapchain for recylcing the underlying allocations and so it's not enough
 * to do buffer management of complete frames.
 */
struct gm_buffer_vtable
{
    void (*free)(struct gm_buffer *self);
    void (*add_breadcrumb)(struct gm_buffer *self,
                           const char *name);
};

struct gm_buffer
{
    _Atomic(int) ref;

    struct gm_buffer_vtable *api;

    /* XXX: currently assuming heap allocated buffers, but probably generalised
     * later.
     *
     * TODO: consider moving state behing buffer->api in case we want a stable
     * ABI.
     */
    size_t len;
    void *data;
};

inline void
gm_buffer_add_breadcrumb(struct gm_buffer *buffer, const char *tag)
{
    buffer->api->add_breadcrumb(buffer, tag);
}

inline struct gm_buffer *
gm_buffer_ref(struct gm_buffer *buffer)
{
    assert(buffer->ref >= 0); // implies use after free!
    gm_buffer_add_breadcrumb(buffer, "ref");
    atomic_fetch_add(&buffer->ref, 1);
    return buffer;
}

inline void
gm_buffer_unref(struct gm_buffer *buffer)
{
    gm_buffer_add_breadcrumb(buffer, "unref");
    if (__builtin_expect(atomic_fetch_sub(&buffer->ref, 1) <= 1, 0))
        buffer->api->free(buffer);
}

/* A reference to an immutable frame comprised of multiple buffers
 *
 * When the frame is no longer needed then gm_frame_unref() should be called to
 * free/recycle the storage when there are no longer any users of the data.
 *
 * This design is intended to abstract an underlying swapchain for recycling
 * the allocations used to hold a frame such that there may be multiple
 * decoupled/unsynchronized consumers of a single frame (such as a rendering
 * thread and an image processing thread).
 *
 * So long as you hold a reference to a frame then it's safe to use the
 * embedded function pointers and safe to read the underlying buffers.
 *
 * Never modify the contents of buffers, make a new frame for modifications if
 * necessary.
 *
 * Aim to release references promptly considing that the production of new
 * frames may eventually become throttled waiting for previous frames to be
 * released.
 */

struct gm_frame;

struct gm_frame_vtable
{
    void (*free)(struct gm_frame *self);
    void (*add_breadcrumb)(struct gm_frame *self,
                           const char *name);
};

struct gm_frame
{
    _Atomic(int) ref;

    struct gm_frame_vtable *api;

    /* TODO: consider putting some of this behind frame->api in case we
     * want a stable ABI.
     */
    uint64_t timestamp;
    enum gm_rotation camera_rotation;

    // Note it's assumed the frame will be rotated according to the camera
    // rotation before interpreting the pose or gravity vector...
    //
    struct gm_pose pose;
    bool gravity_valid;
    float gravity[3];

    struct gm_buffer *depth;
    enum gm_format depth_format; // ignore if depth is NULL
    struct gm_intrinsics depth_intrinsics; // ignore if depth is NULL

    struct gm_buffer *video;
    enum gm_format video_format; // ignore if video is NULL
    struct gm_intrinsics video_intrinsics; // ignore if video is NULL

    /* If true then this frame should not be compared with any
     * previous frames. It might imply a signficant camera
     * re-configuration has happened or a recording may have
     * looped.
     *
     * E.g. This can trigger a reset of any motion tracking state
     */
    bool discontinuity;

    /* Paused frames will toggle on some debug functionality within the viewer
     * and context whereby tracking state for paused frames won't be saved and
     * so it's possible to investigate the effects of various tracking
     * properties while knowing that each re-iteration of the same frame will
     * start with the same state (e.g.  motion based analysis for the paused
     * frame will be repeatable instead of being interpreted like nothing is
     * moving)
     */
    bool paused;
};

inline void
gm_frame_add_breadcrumb(struct gm_frame *frame, const char *tag)
{
    frame->api->add_breadcrumb(frame, tag);
}

inline struct gm_frame *
gm_frame_ref(struct gm_frame *frame)
{
    assert(frame->ref >= 0); // implies use after free!
    gm_frame_add_breadcrumb(frame, "ref");
    atomic_fetch_add(&frame->ref, 1);
    return frame;
}

inline void
gm_frame_unref(struct gm_frame *frame)
{
    gm_frame_add_breadcrumb(frame, "unref");
    if (__builtin_expect(atomic_fetch_sub(&frame->ref, 1) <= 1, 0))
        frame->api->free(frame);
}

struct gm_tracking;

struct gm_tracking_vtable
{
    void (*free)(struct gm_tracking *self);
    void (*add_breadcrumb)(struct gm_tracking *self,
                           const char *name);
};

struct gm_tracking
{
    _Atomic(int) ref;
    struct gm_tracking_vtable *api;
};

inline struct gm_tracking *
gm_tracking_ref(struct gm_tracking *tracking)
{
    atomic_fetch_add(&tracking->ref, 1);
    return tracking;
}

inline void
gm_tracking_unref(struct gm_tracking *tracking)
{
    if (__builtin_expect(atomic_fetch_sub(&tracking->ref, 1) <= 1, 0))
        tracking->api->free(tracking);
}

struct gm_context;

enum gm_joint_semantic
{
    GM_JOINT_UNKNOWN,
    GM_JOINT_HEAD,
    GM_JOINT_NECK,
    GM_JOINT_LEFT_SHOULDER,
    GM_JOINT_RIGHT_SHOULDER,
    GM_JOINT_LEFT_ELBOW,
    GM_JOINT_LEFT_WRIST,
    GM_JOINT_RIGHT_ELBOW,
    GM_JOINT_RIGHT_WRIST,
    GM_JOINT_LEFT_HIP,
    GM_JOINT_LEFT_KNEE,
    GM_JOINT_RIGHT_HIP,
    GM_JOINT_RIGHT_KNEE,
    GM_JOINT_LEFT_ANKLE,
    GM_JOINT_RIGHT_ANKLE,
};

struct gm_joint {
    bool valid;
    float x;
    float y;
    float z;
};

struct gm_bone;

struct gm_skeleton;

struct gm_prediction;

struct gm_prediction_vtable {
    void (*free)(struct gm_prediction *self);
    void (*add_breadcrumb)(struct gm_prediction *self,
                           const char *name);
};

struct gm_prediction {
    _Atomic(int) ref;
    struct gm_prediction_vtable *api;
};

inline struct gm_prediction *
gm_prediction_ref(struct gm_prediction *prediction)
{
    atomic_fetch_add(&prediction->ref, 1);
    return prediction;
}

inline void
gm_prediction_unref(struct gm_prediction *prediction)
{
    if (__builtin_expect(atomic_fetch_sub(&prediction->ref, 1) <= 1, 0))
        prediction->api->free(prediction);
}

#ifdef __cplusplus
extern "C" {
#endif

struct gm_context *gm_context_new(struct gm_logger *logger, char **err);

/* Before starting to feed a context frames from a new device then all buffered
 * state pertaining to the current/previous device can be flushed/cleared out
 * with this api.
 *
 * XXX: It's the user's/caller's responsibility to ensure they have dropped all
 * references to context resources (e.g. tracking and prediction objects)
 * before calling this function.
 */
void gm_context_flush(struct gm_context *ctx, char **err);

/* XXX: It's the user's/caller's responsibility to ensure they have dropped all
 * references to context resources (e.g. tracking and prediction objects)
 * before calling this function.
 */
void gm_context_destroy(struct gm_context *ctx);

struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx);

void
gm_context_set_config(struct gm_context *ctx, JSON_Value *json_config);

int
gm_context_get_n_joints(struct gm_context *ctx);

const char *
gm_context_get_joint_name(struct gm_context *ctx, int joint_id);

const enum gm_joint_semantic
gm_context_get_joint_semantic(struct gm_context *ctx, int joint_id);

void
gm_context_set_max_depth_pixels(struct gm_context *ctx, int max_pixels);

void
gm_context_set_max_video_pixels(struct gm_context *ctx, int max_pixels);

void
gm_context_set_depth_to_video_camera_extrinsics(struct gm_context *ctx,
                                                struct gm_extrinsics *extrinsics);

void
gm_context_rotate_intrinsics(struct gm_context *ctx,
                             const struct gm_intrinsics *intrinsics_in,
                             struct gm_intrinsics *intrinsics_out,
                             enum gm_rotation rotation);

/* Enable skeletal tracking */
void
gm_context_enable(struct gm_context *ctx);

/* Disable skeltal tracking */
void
gm_context_disable(struct gm_context *ctx);

bool
gm_context_notify_frame(struct gm_context *ctx,
                        struct gm_frame *frame);

/* XXX:
 *
 * The current design delivers events as a way to help the receiver remain
 * decoupled from internal design / implementation details, and to try and E.g.
 * keep the gm_device and gm_context layers decoupled from each other. The
 * events are expected to be processed via a mainloop run on a known thread
 * with known locking.
 *
 * It's undefined what thread an event notification is delivered on
 * and undefined what locks may be held by the device/context subsystem
 * (and so reentrancy may result in a dead-lock).
 *
 * Events should not be processed synchronously within notification callbacks
 * and instead work should be queued to run on a known thread with a
 * deterministic state for locks...
 */
void
gm_context_set_event_callback(struct gm_context *ctx,
                              void (*event_callback)(struct gm_context *ctx,
                                                     struct gm_event *event,
                                                     void *user_data),
                              void *user_data);

/* Since events should not be synchronously handled within the above event
 * callback (considering the undefined state) then this API should be used
 * after an event has finally been handled.
 */
void
gm_context_event_free(struct gm_event *event);

/* Should be called every frame from the render thread with a gles context
 * bound to have a chance to use the gpu.
 */
void
gm_context_render_thread_hook(struct gm_context *ctx);

int
gm_context_get_n_stages(struct gm_context *ctx);

const char *
gm_context_get_stage_name(struct gm_context *ctx,
                          int stage);

const char *
gm_context_get_stage_description(struct gm_context *ctx,
                                 int stage);

// We maintain a per-stage circular buffer of duration measurements
// (both across full frames and individual invocations)
//
// This lets us query more stable/aggregated statistics for each
// stage, useful for displaying in a UI.

uint64_t
gm_context_get_stage_frame_duration_avg(struct gm_context *ctx,
                                        int stage);
uint64_t
gm_context_get_stage_frame_duration_median(struct gm_context *ctx,
                                           int stage);
uint64_t
gm_context_get_stage_run_duration_avg(struct gm_context *ctx,
                                      int stage);
uint64_t
gm_context_get_stage_run_duration_median(struct gm_context *ctx,
                                         int stage);

int
gm_context_get_stage_n_images(struct gm_context *ctx,
                              int stage);

const char *
gm_context_get_stage_nth_image_name(struct gm_context *ctx,
                                    int stage,
                                    int n);

const char *
gm_context_get_stage_nth_image_description(struct gm_context *ctx,
                                           int stage,
                                           int n);

struct gm_ui_properties *
gm_context_get_stage_ui_properties(struct gm_context *ctx, int stage);

uint64_t
gm_context_get_average_frame_duration(struct gm_context *ctx);

struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx);

struct gm_prediction *
gm_context_get_prediction(struct gm_context *ctx,
                          uint64_t timestamp);

/* Gets the sum of the square of the difference between min/max bone lengths
 * and actual bone lengths from the inferred skeleton.
 */
float
gm_context_get_skeleton_distance(struct gm_context *ctx,
                                 const struct gm_skeleton *skeleton);

uint64_t
gm_prediction_get_timestamp(struct gm_prediction *prediction);

const struct gm_skeleton *
gm_prediction_get_skeleton(struct gm_prediction *prediction);

const struct gm_intrinsics *
gm_tracking_get_video_camera_intrinsics(struct gm_tracking *tracking);

const struct gm_intrinsics *
gm_tracking_get_depth_camera_intrinsics(struct gm_tracking *tracking);

const float *
gm_tracking_get_label_probabilities(struct gm_tracking *tracking,
                                    int *width,
                                    int *height);

const struct gm_point_rgba *
gm_tracking_get_debug_point_cloud(struct gm_tracking *_tracking,
                                  int *n_points,
                                  struct gm_intrinsics *debug_cloud_intrinsics);

const struct gm_point_rgba *
gm_tracking_get_debug_lines(struct gm_tracking *tracking,
                            int *n_lines);

uint64_t
gm_tracking_get_duration(struct gm_tracking *tracking);

// Total duration for the whole frame, considering that one stage (such
// as label inference) may be run multiple times per-frame
uint64_t
gm_tracking_get_stage_duration(struct gm_tracking *tracking,
                               int stage);
// In case the stage was run multiple times then what was the average
// duration
uint64_t
gm_tracking_get_stage_run_duration_avg(struct gm_tracking *_tracking,
                                       int stage_index);
// In case the stage was run multiple times then what was the median
// duration
uint64_t
gm_tracking_get_stage_run_duration_median(struct gm_tracking *_tracking,
                                          int stage_index);

#if 0
const struct gm_point_rgba *
gm_tracking_get_stage_debug_point_cloud(struct gm_tracking *tracking,
                                        int stage,
                                        int *n_points);

const struct gm_point_rgba *
gm_tracking_get_stage_debug_lines(struct gm_tracking *tracking,
                                  int stage,
                                  int *n_lines);
#endif

bool
gm_tracking_create_stage_rgb_image(struct gm_tracking *tracking,
                                   int stage,
                                   int image_id,
                                   int *width,
                                   int *height,
                                   uint8_t **output);

bool
gm_tracking_has_skeleton(struct gm_tracking *tracking);

const struct gm_skeleton *
gm_tracking_get_skeleton(struct gm_tracking *tracking);

const struct gm_skeleton *
gm_tracking_get_raw_skeleton(struct gm_tracking *tracking);

uint64_t
gm_tracking_get_timestamp(struct gm_tracking *tracking);

bool
gm_tracking_was_successful(struct gm_tracking *tracking);

const struct gm_point_rgba *
gm_tracking_get_pipeline_stage_data(struct gm_tracking *tracking,
                                    int *n_points);

struct gm_skeleton *
gm_skeleton_new(struct gm_context *ctx,
                struct gm_joint *joints);

struct gm_skeleton *
gm_skeleton_new_from_json(struct gm_context *ctx,
                          const char *asset_name);

struct gm_skeleton *
gm_skeleton_resize(struct gm_context *ctx,
                   const struct gm_skeleton *skeleton,
                   const struct gm_skeleton *ref_skeleton,
                   int parent_joint);

int
gm_skeleton_get_n_joints(const struct gm_skeleton *skeleton);

int
gm_skeleton_get_n_bones(const struct gm_skeleton *skeleton);

const struct gm_bone *
gm_skeleton_get_bone(const struct gm_skeleton *skeleton,
                     int bone);

const struct gm_bone *
gm_skeleton_find_bone(const struct gm_skeleton *skeleton,
                      int head,
                      int tail);

const struct gm_joint *
gm_skeleton_get_joint(const struct gm_skeleton *skeleton, int joint);

float
gm_skeleton_compare_angle(const struct gm_skeleton *a,
                          const struct gm_skeleton *b,
                          const struct gm_bone *bone);

float
gm_skeleton_angle_diff_cumulative(const struct gm_skeleton *a,
                                  const struct gm_skeleton *b);

bool
gm_skeleton_save(const struct gm_skeleton *skeleton,
                 const char *filename);

void
gm_skeleton_free(struct gm_skeleton *skeleton);

int
gm_bone_get_head(const struct gm_bone *bone);

int
gm_bone_get_tail(const struct gm_bone *bone);

float
gm_bone_get_length(const struct gm_bone *bone);

void
gm_bone_get_angle(const struct gm_bone *bone,
                  float *out_xyzw);

#ifdef __cplusplus
}
#endif
