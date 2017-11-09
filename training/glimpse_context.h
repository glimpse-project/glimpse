#pragma once

#include <assert.h>
#include <tango_client_api.h>

#include "loader.h"
#include "half.hpp"

enum gm_format {
    GM_FORMAT_UNKNOWN,
    GM_FORMAT_Z_U16_MM,
    GM_FORMAT_Z_F32_M,
    GM_FORMAT_Z_F16_M,
    GM_FORMAT_LUMINANCE_U8,
};

inline int __attribute__((unused))
gm_format_bytes_per_pixel(enum gm_format format)
{
    switch (format) {
    case GM_FORMAT_UNKNOWN:
        assert(0);
        return 0;
    case GM_FORMAT_LUMINANCE_U8:
        return 1;
    case GM_FORMAT_Z_U16_MM:
    case GM_FORMAT_Z_F16_M:
        return 2;
    case GM_FORMAT_Z_F32_M:
        return 4;
    }

    assert(0);
    return 0;
}

enum gm_property_type {
    GM_PROPERTY_INT,
    GM_PROPERTY_FLOAT,
    GM_PROPERTY_FLOAT_VEC3,
};

struct gm_ui_property {
    const char *name;
    const char *desc;
    enum gm_property_type type;
    union {
        int *int_ptr;
        float *float_ptr;
        float *float_vec3;
    };
    float min;
    float max;
    bool read_only;
};

/* During development and testing it's convenient to have direct tuneables
 * we can play with at runtime...
 */
struct gm_ui_properties {
    pthread_mutex_t lock;
    int n_properties;
    struct gm_ui_property *properties;
};

enum gm_event_type
{
    GM_EVENT_REQUEST_FRAME,
    GM_EVENT_TRACKING_READY
};

#define GM_REQUEST_FRAME_DEPTH      1<<0
#define GM_REQUEST_FRAME_LUMINANCE  1<<1

struct gm_event
{
    enum gm_event_type type;

    union {
        struct {
            uint64_t flags;
        } request_frame;
    };
};


/* FIXME put gm_frame in a glimpse_internal.h. These should be
 * exposed to glimpse_cameras.c but not to apps/middleware
 */

struct gm_frame
{
    uint64_t timestamp; // CLOCK_MONOTONIC
    enum gm_format depth_format;
    void *depth;
    enum gm_format video_format;
    void *video;

    //TODO
#if 0
    enum gm_rotation rotation;
    float down[3];
#endif
};


struct gm_context;

#ifdef __cplusplus
extern "C" {
#endif

struct gm_context *gm_context_new(char **err);
void gm_context_destroy(struct gm_context *ctx);


struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx);


/* XXX: the timebase of the timestamp is currently undefined with units
 * of nanoseconds.
 * XXX: it can currently be assumed that there's an implicit copy made
 * of the luminance data and the given luminance buffer is not required
 * once the call returns.
 * XXX: we don't say whether the image is rectified or not.
 * XXX: this wouldn't be called if using GL to downsample the luminance data
 * via a yuv texture-external sampler.
 * XXX: we currently make no assumption about what thread this is called
 * on, in case camera/device IO is associated with a dedicated thread.
 */
void
gm_context_update_luminance(struct gm_context *ctx,
                            int width,
                            int height,
                            uint8_t *luminance,
                            uint64_t timestamp);

/* TODO: make tango api agnostic */
void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       TangoCameraIntrinsics *intrinsics);

void
gm_context_set_video_camera_intrinsics(struct gm_context *ctx,
                                       TangoCameraIntrinsics *intrinsics);

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


struct gm_tracking;

struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx);

void *
gm_frame_get_video_buffer(struct gm_frame *frame);

enum gm_format
gm_frame_get_video_format(struct gm_frame *frame);

/* XXX: not really a good approach since you can't fetch the latest state
 * atomically...
 */

const float *
gm_tracking_get_label_probabilities(struct gm_tracking *tracking,
                                    int *width,
                                    int *height);

const uint8_t *
gm_tracking_get_label_map(struct gm_tracking *tracking,
                          int *width,
                          int *height);

const uint8_t *
gm_tracking_get_rgb_label_map(struct gm_tracking *tracking,
                              int *width,
                              int *height);

const uint8_t *
gm_tracking_get_rgb_depth(struct gm_tracking *tracking);

#ifdef __cplusplus
}
#endif
