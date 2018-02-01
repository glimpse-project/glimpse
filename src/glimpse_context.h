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

#include "glimpse_config.h"
#include "loader.h"

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
};


enum gm_event_type
{
    GM_EVENT_REQUEST_FRAME,
    GM_EVENT_TRACKING_READY
};

#define GM_REQUEST_FRAME_DEPTH  1<<0
#define GM_REQUEST_FRAME_VIDEO  1<<1

struct gm_event
{
    enum gm_event_type type;

    union {
        struct {
            uint64_t flags;
        } request_frame;
    };
};

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
};

struct gm_buffer
{
    int ref;

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

inline struct gm_buffer *
gm_buffer_ref(struct gm_buffer *buffer)
{
    buffer->ref++;
    return buffer;
}

inline void
gm_buffer_unref(struct gm_buffer *buffer)
{
    if (__builtin_expect(--(buffer->ref) < 1, 0))
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

struct gm_frame_vtable
{
    void (*free)(struct gm_frame *self);
};

struct gm_frame
{
    int ref;

    struct gm_frame_vtable *api;

    /* TODO: consider putting some of this behind frame->api in case we
     * want a stable ABI.
     */
    uint64_t timestamp;
    enum gm_format depth_format;
    struct gm_buffer *depth;
    enum gm_format video_format;
    struct gm_buffer *video;
};

inline struct gm_frame *
gm_frame_ref(struct gm_frame *frame)
{
    frame->ref++;
    return frame;
}

inline void
gm_frame_unref(struct gm_frame *frame)
{
    if (__builtin_expect(--(frame->ref) < 1, 0))
        frame->api->free(frame);
}

struct gm_tracking_vtable
{
    void (*free)(struct gm_tracking *self);
};

struct gm_tracking
{
    int ref;
    struct gm_tracking_vtable *api;
};

inline struct gm_tracking *
gm_tracking_ref(struct gm_tracking *tracking)
{
    tracking->ref++;
    return tracking;
}

inline void
gm_tracking_unref(struct gm_tracking *tracking)
{
    if (__builtin_expect(--(tracking->ref) < 1, 0))
        tracking->api->free(tracking);
}

struct gm_intrinsics {
  uint32_t width;
  uint32_t height;

  double fx;
  double fy;
  double cx;
  double cy;

  /* TODO: add distortion model description */
};

struct gm_extrinsics {
  float rotation[9];    // Column-major 3x3 rotation matrix
  float translation[3]; // Translation vector, in meters
};

struct gm_context;

typedef struct {
    float x;
    float y;
    float z;
    uint32_t rgba;
} GlimpsePointXYZRGBA;

#ifdef __cplusplus
extern "C" {
#endif

struct gm_context *gm_context_new(struct gm_logger *logger, char **err);
void gm_context_destroy(struct gm_context *ctx);


struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx);

void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics);

void
gm_context_set_video_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics);

void
gm_context_set_depth_to_video_camera_extrinsics(struct gm_context *ctx,
                                                struct gm_extrinsics *extrinsics);

const gm_intrinsics *
gm_context_get_training_intrinsics(struct gm_context *ctx);

/* Enable skeletal tracking */
void
gm_context_enable(struct gm_context *ctx);

/* Disable skeltal tracking */
void
gm_context_disable(struct gm_context *ctx);

bool
gm_context_notify_frame(struct gm_context *ctx,
                        struct gm_frame *frame);

void
gm_context_set_event_callback(struct gm_context *ctx,
                              void (*event_callback)(struct gm_context *ctx,
                                                     struct gm_event *event,
                                                     void *user_data),
                              void *user_data);

void
gm_context_event_free(struct gm_event *event);

/* Should be called every frame from the render thread with a gles context
 * bound to have a chance to use the gpu.
 */
void
gm_context_render_thread_hook(struct gm_context *ctx);


struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx);

/* XXX: not really a good approach since you can't fetch the latest state
 * atomically...
 */

const float *
gm_tracking_get_label_probabilities(struct gm_tracking *tracking,
                                    int *width,
                                    int *height);

const uint8_t *
gm_tracking_get_rgb_label_map(struct gm_tracking *tracking,
                              int *width,
                              int *height);

const uint8_t *
gm_tracking_get_rgb_depth(struct gm_tracking *tracking);

const float *
gm_tracking_get_depth(struct gm_tracking *tracking);

const uint32_t *
gm_tracking_get_video(struct gm_tracking *tracking);

const float *
gm_tracking_get_joint_positions(struct gm_tracking *tracking,
                                int *n_joints);
#ifdef __cplusplus
}
#endif
