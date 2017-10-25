#pragma once

#include <tango_client_api.h>

#include "loader.h"
#include "half.hpp"

struct gm_context;

enum gm_property_type {
    GM_PROPERTY_INT,
    GM_PROPERTY_FLOAT,
};

struct gm_ui_property {
    const char *name;
    const char *desc;
    enum gm_property_type type;
    union {
        int *int_ptr;
        float *float_ptr;
    };
    float min;
    float max;
};

/* During development and testing it's convenient to have direct tuneables
 * we can play with at runtime...
 */
struct gm_ui_properties {
    pthread_mutex_t lock;
    int n_properties;
    struct gm_ui_property *properties;
};

struct gm_context *gm_context_new(char **err);
void gm_context_destroy(struct gm_context *ctx);

struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx);


/* XXX: the timebase of the timestamp is currently undefined with units
 * of seconds.
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
                            double timestamp,
                            int width, int height,
                            uint8_t *luminance);

/* TODO: make tango api agnostic */
void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       TangoCameraIntrinsics *intrinsics);


/* XXX: the timebase of the timestamp is currently undefined with units
 * of seconds.
 * XXX: if can currently be assume that there's an implici copy made of
 * the depth data and the given depth buffer is not required once the call
 * returns.
 * XXX: we don't say whether the image is rectified or not.
 * XXX: this wouldn't be called if using GL to downsample the luminance data
 * via a yuv texture-external sampler.
 */
void
gm_context_update_depth_from_u16_mm(struct gm_context *ctx,
                                    double timestamp,
                                    int width, int height,
                                    uint16_t *depth);

/* Should be called every frame from the render thread with a gles context
 * bound to have a chance to use the gpu.
 */
void
gm_context_render_thread_hook(struct gm_context *ctx);


/* XXX: not really a good approach since you can't fetch the latest state
 * atomically...
 */

/* width and height implicitly matches depth camera intrinsics */
const float *
gm_context_get_latest_label_probabilities(struct gm_context *ctx,
                                          int *width,
                                          int *height);

/* width and height implicitly matches depth camera intrinsics */
const uint8_t *
gm_context_get_latest_label_map(struct gm_context *ctx,
                                int *width,
                                int *height);

const uint8_t *
gm_context_get_latest_rgb_label_map(struct gm_context *ctx,
                                    int *width,
                                    int *height);
