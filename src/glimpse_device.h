#pragma once

#include "glimpse_context.h"

enum gm_device_event_type
{
    GM_DEV_EVENT_FRAME_READY
};

struct gm_device_event
{
    enum gm_device_event_type type;

    union {
        struct {
            uint64_t met_requirements;
        } frame_ready;
    };
};


enum gm_device_type {
    GM_DEVICE_KINECT,
    GM_DEVICE_RECORDING,
};

struct gm_device_config {
    enum gm_device_type type;
    union {
        struct {
            int device_number;
        } kinect;
        struct {
            const char *path;
        } recording;
    };
};

#ifdef __cplusplus
extern "C" {
#endif

struct gm_device *
gm_device_open(struct gm_logger *log,
               struct gm_device_config *config,
               char **err);

struct gm_ui_properties *
gm_device_get_ui_properties(struct gm_device *dev);

struct gm_intrinsics *
gm_device_get_depth_intrinsics(struct gm_device *dev);

struct gm_intrinsics *
gm_device_get_video_intrinsics(struct gm_device *dev);

void
gm_device_set_event_callback(struct gm_device *dev,
                             void (*event_callback)(struct gm_device *dev,
                                                    struct gm_device_event *event,
                                                    void *user_data),
                             void *user_data);

/* It's expected that events aren't synchronously handed within the above
 * event callback considering that it's undefined what thread the callback
 * is invoked on and it's undefined what locks might be held during the
 * invocation whereby the device api may not be reentrant at that point.
 *
 * An event will likely be queued for processing later but when processing
 * is finished then the event structure needs to be freed with this api:
 */
void gm_device_event_free(struct gm_device_event *event);

void gm_device_start(struct gm_device *dev);
// TODO add _stop() api also

void gm_device_close(struct gm_device *dev);

void
gm_device_request_frame(struct gm_device *dev, uint64_t requirements);

struct gm_frame *
gm_device_get_latest_frame(struct gm_device *dev);

void
gm_device_free_frame(struct gm_device *dev,
                     struct gm_frame *frame);

#ifdef __cplusplus
}
#endif
