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
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

#ifdef USE_FREENECT
#include <libfreenect.h>
#endif

#include "half.hpp"
#include "xalloc.h"

#include "image_utils.h"

#include "glimpse_log.h"
#include "glimpse_device.h"

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)


using half_float::half;

struct gm_mem_pool {
    struct gm_device *dev;

    const char *name;

    pthread_mutex_t lock;
    pthread_cond_t available_cond;
    unsigned max_size;
    std::vector<void *> available;
    std::vector<void *> busy;

    void *(*alloc_mem)(struct gm_mem_pool *pool);
    void (*free_mem)(struct gm_mem_pool *pool, void *mem);
};

struct gm_device_buffer
{
    struct gm_buffer base;

    struct gm_buffer_vtable vtable;

    struct gm_device *dev;
    struct gm_mem_pool *pool;
};

struct gm_device_frame
{
    struct gm_frame base;

    struct gm_frame_vtable vtable;

    struct gm_device *dev;
    struct gm_mem_pool *pool;

    //TODO
#if 0
    enum gm_rotation rotation;
    float down[3];
#endif
};


struct gm_device
{
    enum gm_device_type type;

    struct gm_logger *log;

    union {
        struct {
            int frame;
            uint64_t time;

            int n_images;
            half **depth_images;
            uint8_t **lum_images;

            pthread_t io_thread;
        } dummy;

#ifdef USE_FREENECT
        struct {
            freenect_context *fctx;
            freenect_device *fdev;

            int ir_brightness;
            float req_tilt; // tilt requested via UI
            float phys_tilt; // tilt currently reported by HW
            float accel[3];
            float mks_accel[3];
            pthread_t io_thread;
        } kinect;
#endif
    };

    struct gm_intrinsics video_camera_intrinsics;
    struct gm_intrinsics depth_camera_intrinsics;
    struct gm_extrinsics depth_to_video_extrinsics;

    void (*frame_callback)(struct gm_device *dev,
                           struct gm_frame *frame,
                           void *user_data);
    void *frame_callback_data;

    /* What data is required for the next frame?
     * E.g. _DEPTH | _VIDEO
     */
    uint64_t frame_request_requirements;

    pthread_mutex_t swap_buffers_lock;

    enum gm_format depth_format;

    enum gm_format video_format;

    /* Here 'ready' buffers are one that are ready to be collected into a
     * frame if requested. The 'back' buffers are the ones that the hardware
     * is currently writing into.
     */
    struct gm_mem_pool video_buf_pool;
    struct gm_device_buffer *video_buf_ready;
    struct gm_device_buffer *video_buf_back;

    struct gm_mem_pool depth_buf_pool;
    struct gm_device_buffer *depth_buf_ready;
    struct gm_device_buffer *depth_buf_back;

    uint64_t frame_time;

    struct gm_mem_pool frame_pool;
    struct gm_frame *last_frame;

    /* If depth_mid buffer is valid then corresponding _DEPTH bit is set */
    uint64_t frame_ready_requirements;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;

    void (*event_callback)(struct gm_device *dev,
                           struct gm_device_event *event,
                           void *user_data);

    void *callback_data;

    pthread_mutex_t request_requirements_lock;
};

static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static struct gm_device_event *
device_event_alloc(enum gm_device_event_type type)
{
    struct gm_device_event *event =
        (struct gm_device_event *)xcalloc(sizeof(struct gm_device_event), 1);

    event->type = type;

    return event;
}

void
gm_device_event_free(struct gm_device_event *event)
{
    free(event);
}

static void *
mem_pool_acquire_resource(struct gm_mem_pool *pool)
{
    struct gm_device *dev = pool->dev;
    void *resource;

    pthread_mutex_lock(&pool->lock);

    /* Sanity check with arbitrary upper limit for the number of allocations */
    gm_assert(dev->log,
              (pool->busy.size() + pool->available.size()) < 100,
              "'%s' memory pool growing out of control (%lu allocations)",
              pool->name,
              (pool->busy.size() + pool->available.size()));

    if (pool->available.size()) {
        resource = pool->available.back();
        pool->available.pop_back();
    } else if (pool->busy.size() + pool->available.size() > pool->max_size) {

        gm_debug(dev->log,
                 "Throttling \"%s\" pool acquisition, waiting for old %s object to be released\n",
                 pool->name, pool->name);

        while (!pool->available.size())
            pthread_cond_wait(&pool->available_cond, &pool->lock);

        resource = pool->available.back();
        pool->available.pop_back();
    } else {
        resource = pool->alloc_mem(pool);
    }

    pool->busy.push_back(resource);

    pthread_mutex_unlock(&pool->lock);

    return resource;
}

static void
mem_pool_recycle_resource(struct gm_mem_pool *pool, void *resource)
{
    pthread_mutex_lock(&pool->lock);

    unsigned size = pool->busy.size();
    for (unsigned i = 0; i < size; i++) {
        if (pool->busy[i] == resource) {
            pool->busy[i] = pool->busy.back();
            pool->busy.pop_back();
            break;
        }
    }

    gm_assert(pool->dev->log,
              pool->busy.size() == (size - 1),
              "Didn't find recycled resource %p in %s pool's busy list",
              resource,
              pool->name);

    pool->available.push_back(resource);
    pthread_cond_broadcast(&pool->available_cond);
    pthread_mutex_unlock(&pool->lock);
}

static void
mem_pool_free_resources(struct gm_mem_pool *pool)
{
    gm_assert(pool->dev->log,
              pool->busy.size() == 0,
              "Shouldn't be freeing a pool with resources still in use");

    while (pool->available.size()) {
        void *resource = pool->available.back();
        pool->available.pop_back();
        pool->free_mem(pool, resource);
    }
}

static struct gm_device_frame *
mem_pool_acquire_frame(struct gm_mem_pool *pool)
{
    struct gm_device_frame *frame = (struct gm_device_frame *)
        mem_pool_acquire_resource(pool);
    frame->base.ref = 1;
    return frame;
}

static struct gm_device_buffer *
mem_pool_acquire_buffer(struct gm_mem_pool *pool)
{
    struct gm_device_buffer *buffer = (struct gm_device_buffer *)
        mem_pool_acquire_resource(pool);
    buffer->base.ref = 1;
    return buffer;
}

static void
device_frame_recycle(struct gm_frame *self)
{
    struct gm_device_frame *frame = (struct gm_device_frame *)self;
    struct gm_device *dev = frame->dev;
    struct gm_mem_pool *pool = frame->pool;

    gm_assert(frame->dev->log, frame->base.ref == 0, "Unbalanced frame unref");

    mem_pool_recycle_resource(&dev->video_buf_pool, self->video);
    mem_pool_recycle_resource(&dev->depth_buf_pool, self->depth);
    mem_pool_recycle_resource(pool, frame);
}

static void
device_frame_free(struct gm_mem_pool *pool, void *resource)
{
    struct gm_device_frame *frame = (struct gm_device_frame *)resource;

    xfree(frame);
}

static void *
device_frame_alloc(struct gm_mem_pool *pool)
{
    struct gm_device *dev = pool->dev;
    struct gm_device_frame *frame =
        (struct gm_device_frame *)xcalloc(sizeof(*frame), 1);

    frame->vtable.free = device_frame_recycle;
    frame->dev = dev;
    frame->pool = pool;

    frame->base.ref = 1;
    frame->base.api = &frame->vtable;

    return frame;
}

static void
device_buffer_recycle(struct gm_buffer *self)
{
    struct gm_device_buffer *buf = (struct gm_device_buffer *)self;
    struct gm_mem_pool *pool = buf->pool;

    gm_assert(buf->dev->log, buf->base.ref == 0, "Unbalanced buffer unref");

    mem_pool_recycle_resource(pool, buf);
}

static void *
device_video_buf_alloc(struct gm_mem_pool *pool)
{
    struct gm_device *dev = pool->dev;
    struct gm_device_buffer *buf =
        (struct gm_device_buffer *)xcalloc(sizeof(*buf), 1);

    buf->vtable.free = device_buffer_recycle;
    buf->dev = dev;
    buf->pool = pool;

    buf->base.ref = 1;
    buf->base.api = &buf->vtable;

    /* allocated large enough for _RGB format */
    int video_width = dev->video_camera_intrinsics.width;
    int video_height = dev->video_camera_intrinsics.height;
    buf->base.len = video_width * video_height * 3;
    buf->base.data = xmalloc(buf->base.len);

    return buf;
}

static void
device_buffer_free(struct gm_mem_pool *pool, void *resource)
{
    struct gm_buffer *buf = (struct gm_buffer *)resource;
    xfree(buf);
}

static void *
device_depth_buf_alloc(struct gm_mem_pool *pool)
{
    struct gm_device *dev = pool->dev;
    struct gm_device_buffer *buf =
        (struct gm_device_buffer *)xcalloc(sizeof(*buf), 1);

    buf->vtable.free = device_buffer_recycle;
    buf->dev = dev;
    buf->pool = pool;

    buf->base.ref = 1;
    buf->base.api = &buf->vtable;

    /* Allocated large enough got _U16_MM data */
    int depth_width = dev->depth_camera_intrinsics.width;
    int depth_height = dev->depth_camera_intrinsics.height;
    buf->base.len = depth_width * depth_height * 2;
    buf->base.data = xmalloc(buf->base.len);

    return buf;
}

/* XXX: the request_requirements_lock must be held while calling this.
 *
 * Note: this implies that it's not currently safe for the reciever of the
 * event to synchronously request a new frame or call any device api that
 * might affect these requirements (needing the same lock)
 */
static void
notify_frame_locked(struct gm_device *dev)
{
    struct gm_device_event *event = device_event_alloc(GM_DEV_EVENT_FRAME_READY);

    event->frame_ready.met_requirements = dev->frame_request_requirements;
    dev->frame_request_requirements = 0;

    dev->event_callback(dev, event, dev->callback_data);
}

#ifdef USE_FREENECT
static void
kinect_depth_frame_cb(freenect_device *fdev, void *depth, uint32_t timestamp)
{
    struct gm_device *dev = (struct gm_device *)freenect_get_user(fdev);

    if (!(dev->frame_request_requirements & GM_REQUEST_FRAME_DEPTH))
        return;

    pthread_mutex_lock(&dev->swap_buffers_lock);

    struct gm_device_buffer *old = dev->depth_buf_ready;
    dev->depth_buf_ready = dev->depth_buf_back;
    dev->depth_buf_back = mem_pool_acquire_buffer(&dev->depth_buf_pool);
    dev->frame_time = (uint64_t)timestamp;
    dev->frame_ready_requirements |= GM_REQUEST_FRAME_VIDEO;

    freenect_set_depth_buffer(fdev, dev->depth_buf_back->base.data);
    if (old)
        gm_buffer_unref(&old->base);

    pthread_mutex_unlock(&dev->swap_buffers_lock);

    pthread_mutex_lock(&dev->request_requirements_lock);
    if ((dev->frame_request_requirements & dev->frame_ready_requirements) ==
        dev->frame_request_requirements)
    {
        notify_frame_locked(dev);
    }
    pthread_mutex_unlock(&dev->request_requirements_lock);
}

static void
kinect_rgb_frame_cb(freenect_device *fdev, void *video, uint32_t timestamp)
{
    struct gm_device *dev = (struct gm_device *)freenect_get_user(fdev);

    if (!(dev->frame_request_requirements & GM_REQUEST_FRAME_VIDEO))
        return;

    pthread_mutex_lock(&dev->swap_buffers_lock);

    struct gm_device_buffer *old = dev->video_buf_ready;
    dev->video_buf_ready = dev->video_buf_back;
    dev->video_buf_back = mem_pool_acquire_buffer(&dev->video_buf_pool);
    dev->frame_time = (uint64_t)timestamp;
    dev->frame_ready_requirements |= GM_REQUEST_FRAME_DEPTH;

    freenect_set_video_buffer(fdev, dev->video_buf_back->base.data);
    if (old)
        gm_buffer_unref(&old->base);

    pthread_mutex_unlock(&dev->swap_buffers_lock);

    pthread_mutex_lock(&dev->request_requirements_lock);
    if ((dev->frame_request_requirements & dev->frame_ready_requirements) ==
        dev->frame_request_requirements)
    {
        notify_frame_locked(dev);
    }
    pthread_mutex_unlock(&dev->request_requirements_lock);
}

static bool
kinect_open(struct gm_device *dev, struct gm_device_config *config, char **err)
{
    if (freenect_init(&dev->kinect.fctx, NULL) < 0) {
        xasprintf(err, "Failed to init libfreenect\n");
        return false;
    }

    /* We get loads of 'errors' from the kinect but it seems to vaguely
     * be working :)
     */
    freenect_set_log_level(dev->kinect.fctx, FREENECT_LOG_FATAL);
    freenect_select_subdevices(dev->kinect.fctx,
                               (freenect_device_flags)(FREENECT_DEVICE_MOTOR |
                                                       FREENECT_DEVICE_CAMERA));

    if (!freenect_num_devices(dev->kinect.fctx)) {
        xasprintf(err, "Failed to find a Kinect device\n");
        freenect_shutdown(dev->kinect.fctx);
        return false;
    }

    if (freenect_open_device(dev->kinect.fctx, &dev->kinect.fdev, 0) < 0) {
        xasprintf(err, "Could not open Kinect device\n");
        freenect_shutdown(dev->kinect.fctx);
        return false;
    }

    freenect_set_user(dev->kinect.fdev, dev);

    dev->kinect.ir_brightness = freenect_get_ir_brightness(dev->kinect.fdev);

    freenect_raw_tilt_state *tilt_state;
    freenect_update_tilt_state(dev->kinect.fdev);
    tilt_state = freenect_get_tilt_state(dev->kinect.fdev);

    dev->kinect.phys_tilt = freenect_get_tilt_degs(tilt_state);
    dev->kinect.req_tilt = dev->kinect.phys_tilt;

    /* libfreenect doesn't give us a way to query camera intrinsics so just
     * using these random/plausible intrinsics found on the internet to avoid
     * manually calibrating for now :)
     */
    dev->depth_camera_intrinsics.width = 640;
    dev->depth_camera_intrinsics.height = 480;
    dev->depth_camera_intrinsics.cx = 339.30780975300314;
    dev->depth_camera_intrinsics.cy = 242.73913761751615;
    dev->depth_camera_intrinsics.fx = 594.21434211923247;
    dev->depth_camera_intrinsics.fy = 591.04053696870778;
    dev->depth_format = GM_FORMAT_Z_U16_MM;

    /* We're going to use Freenect's registered depth mode, which transforms
     * depth to video space, so we don't need video intrinsics/extrinsics.
     */
    dev->video_format = GM_FORMAT_RGB;
    dev->video_camera_intrinsics = dev->depth_camera_intrinsics;
    dev->depth_to_video_extrinsics.rotation[0] = 1.f;
    dev->depth_to_video_extrinsics.rotation[1] = 0.f;
    dev->depth_to_video_extrinsics.rotation[2] = 0.f;
    dev->depth_to_video_extrinsics.rotation[3] = 0.f;
    dev->depth_to_video_extrinsics.rotation[4] = 1.f;
    dev->depth_to_video_extrinsics.rotation[5] = 0.f;
    dev->depth_to_video_extrinsics.rotation[6] = 0.f;
    dev->depth_to_video_extrinsics.rotation[7] = 0.f;
    dev->depth_to_video_extrinsics.rotation[8] = 1.f;

    dev->depth_to_video_extrinsics.translation[0] = 0.f;
    dev->depth_to_video_extrinsics.translation[1] = 0.f;
    dev->depth_to_video_extrinsics.translation[2] = 0.f;

    /* Alternative video intrinsics/extrinsics when not using registered mode.
     * Note, these unfortunately don't actually work.
     */
#if 0
    dev->video_camera_intrinsics.width = 640;
    dev->video_camera_intrinsics.height = 480;
    dev->video_camera_intrinsics.cx = 328.94272028759258;
    dev->video_camera_intrinsics.cy = 267.48068171871557;
    dev->video_camera_intrinsics.fx = 529.21508098293293;
    dev->video_camera_intrinsics.fy = 525.56393630057437;

    dev->depth_to_video_extrinsics.rotation[0] = 0.99984628826577793;
    dev->depth_to_video_extrinsics.rotation[1] = 0.0012635359098409581;
    dev->depth_to_video_extrinsics.rotation[2] = -0.017487233004436643;
    dev->depth_to_video_extrinsics.rotation[3] = -0.0014779096108364480;
    dev->depth_to_video_extrinsics.rotation[4] = 0.99992385683542895;
    dev->depth_to_video_extrinsics.rotation[5] = -0.012251380107679535;
    dev->depth_to_video_extrinsics.rotation[6] = 0.017470421412464927;
    dev->depth_to_video_extrinsics.rotation[7] = 0.012275341476520762;
    dev->depth_to_video_extrinsics.rotation[8] = 0.99977202419716948;

    dev->depth_to_video_extrinsics.translation[0] = 0.019985242312092553;
    dev->depth_to_video_extrinsics.translation[1] = -0.00074423738761617583;
    dev->depth_to_video_extrinsics.translation[2] = -0.010916736334336222;
#endif

    dev->video_camera_intrinsics = dev->depth_camera_intrinsics;

    /* Some alternative intrinsics
     *
     * TODO: we should allow explicit calibrarion and loading these at runtime
     */
#if 0
    dev->depth_camera_intrinsics.cx = 322.515987;
    dev->depth_camera_intrinsics.cy = 259.055966;
    dev->depth_camera_intrinsics.fx = 521.179233;
    dev->depth_camera_intrinsics.fy = 493.033034;

#endif

    freenect_set_video_callback(dev->kinect.fdev, kinect_rgb_frame_cb);
    freenect_set_video_mode(dev->kinect.fdev,
                            freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM,
                                                     FREENECT_VIDEO_RGB));
    dev->video_buf_back = mem_pool_acquire_buffer(&dev->video_buf_pool);
    freenect_set_video_buffer(dev->kinect.fdev, dev->video_buf_back->base.data);

    freenect_set_depth_callback(dev->kinect.fdev, kinect_depth_frame_cb);
    freenect_set_depth_mode(dev->kinect.fdev,
                            freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,
                            FREENECT_DEPTH_REGISTERED)); // MM, aligned to RGB
    /*freenect_set_depth_mode(dev->kinect.fdev,
                            freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,
                                                     FREENECT_DEPTH_MM));*/
    dev->depth_buf_back = mem_pool_acquire_buffer(&dev->depth_buf_pool);
    freenect_set_depth_buffer(dev->kinect.fdev, dev->depth_buf_back->base.data);


    struct gm_ui_property prop;

    prop = gm_ui_property();
    prop.object = dev;
    prop.name = "ir_brightness";
    prop.desc = "IR Brightness";
    prop.type = GM_PROPERTY_INT;
    prop.int_state.ptr = &dev->kinect.ir_brightness;
    prop.int_state.min = 0;
    prop.int_state.max = 50;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = dev;
    prop.name = "request_tilt";
    prop.desc = "Requested Tilt";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &dev->kinect.req_tilt;
    prop.float_state.min = -30;
    prop.float_state.max = 30;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = dev;
    prop.name = "physical tilt";
    prop.desc = "Current Physical Tilt";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &dev->kinect.phys_tilt;
    prop.read_only = true;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = dev;
    prop.name = "accel";
    prop.desc = "Accel";
    prop.type = GM_PROPERTY_FLOAT_VEC3;
    prop.vec3_state.ptr = dev->kinect.accel;
    prop.vec3_state.components[0] = "x";
    prop.vec3_state.components[1] = "y";
    prop.vec3_state.components[2] = "z";
    prop.read_only = true;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.object = dev;
    prop.name = "mks_accel";
    prop.desc = "MKS Accel";
    prop.type = GM_PROPERTY_FLOAT_VEC3;
    prop.vec3_state.ptr = dev->kinect.mks_accel;
    prop.vec3_state.components[0] = "x";
    prop.vec3_state.components[1] = "y";
    prop.vec3_state.components[2] = "z";
    prop.read_only = true;
    dev->properties.push_back(prop);

    dev->properties_state.n_properties = dev->properties.size();
    pthread_mutex_init(&dev->properties_state.lock, NULL);
    dev->properties_state.properties = &dev->properties[0];

    return true;
}

static void
kinect_close(struct gm_device *dev)
{
    freenect_stop_depth(dev->kinect.fdev);
    freenect_stop_video(dev->kinect.fdev);

    freenect_close_device(dev->kinect.fdev);
    freenect_shutdown(dev->kinect.fctx);

    mem_pool_free_resources(&dev->depth_buf_pool);
    mem_pool_free_resources(&dev->video_buf_pool);
    mem_pool_free_resources(&dev->frame_pool);
}

static void *
kinect_io_thread_cb(void *data)
{
    struct gm_device *dev = (struct gm_device *)data;
    int state_check_throttle = 0;

    freenect_set_tilt_degs(dev->kinect.fdev, 0);
    freenect_set_led(dev->kinect.fdev, LED_RED);

    freenect_start_depth(dev->kinect.fdev);
    freenect_start_video(dev->kinect.fdev);

    while (freenect_process_events(dev->kinect.fctx) >= 0) {
        if (state_check_throttle++ >= 2000) {
            freenect_raw_tilt_state* state;
            freenect_update_tilt_state(dev->kinect.fdev);
            state = freenect_get_tilt_state(dev->kinect.fdev);

            dev->kinect.accel[0] = state->accelerometer_x;
            dev->kinect.accel[1] = state->accelerometer_y;
            dev->kinect.accel[2] = state->accelerometer_z;

            double mks_dx, mks_dy, mks_dz;
            freenect_get_mks_accel(state, &mks_dx, &mks_dy, &mks_dz);

            dev->kinect.mks_accel[0] = mks_dx;
            dev->kinect.mks_accel[1] = mks_dy;
            dev->kinect.mks_accel[2] = mks_dz;

            dev->kinect.phys_tilt = freenect_get_tilt_degs(state);
            if (dev->kinect.phys_tilt != dev->kinect.req_tilt) {
                freenect_set_tilt_degs(dev->kinect.fdev, dev->kinect.req_tilt);
            }

            int brightness = freenect_get_ir_brightness(dev->kinect.fdev);
            if (brightness != dev->kinect.ir_brightness) {
                freenect_set_ir_brightness(dev->kinect.fdev,
                                           dev->kinect.ir_brightness);
            }

            state_check_throttle = 0;
        }
    }

    return NULL;
}

static void
kinect_start(struct gm_device *dev)
{
    pthread_create(&dev->kinect.io_thread,
                   NULL, //attributes
                   kinect_io_thread_cb,
                   dev); //data
}
#endif // USE_FREENECT

static bool
directory_recurse(const char *path, const char *ext,
                  std::vector<char *> &files,
                  char **err)
{
    struct dirent *entry;
    struct stat st;
    size_t ext_len;
    char *cur_ext;
    DIR *dir;
    bool ret = true;

    if (!(dir = opendir(path))) {
        xasprintf(err, "Failed to open directory %s\n", path);
        return false;
    }

    ext_len = strlen(ext);

    while ((entry = readdir(dir)) != NULL) {
        char next_path[1024];

        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0)
            continue;

        xsnprintf(next_path, sizeof(next_path), "%s/%s", path, entry->d_name);

        stat(next_path, &st);
        if (S_ISDIR(st.st_mode)) {
            if (!directory_recurse(next_path, ext, files, err)) {
                ret = false;
                break;
            }
        } else if ((cur_ext = strstr(entry->d_name, ext)) &&
                   cur_ext[ext_len] == '\0') {
            files.push_back(strdup(next_path));
        }
    }

    closedir(dir);

    return ret;
}

static bool
recording_open(struct gm_device *dev,
               struct gm_device_config *config, char **err)
{
    /* Load dummy images instead of using Kinect */
    std::vector<char *> exr_files;
    std::vector<char *> png_files;

    if (!directory_recurse(config->recording.path, ".exr", exr_files, err))
        return false;
    if (!directory_recurse(config->recording.path, ".png", png_files, err))
        return false;

    if (exr_files.size() == 0 || png_files.size() == 0) {
        xasprintf(err, "No exr or png files found\n");
        return false;
    }
    if (exr_files.size() != png_files.size()) {
        xasprintf(err, "exr/png quantity mismatch\n");
        return false;
    }

    dev->dummy.n_images = exr_files.size();
    dev->dummy.depth_images = (half**)calloc(sizeof(half*), dev->dummy.n_images);
    dev->dummy.lum_images = (uint8_t**)calloc(sizeof(uint8_t*), dev->dummy.n_images);

    int depth_width = 0;
    int depth_height = 0;
    for (unsigned i = 0; i < exr_files.size(); i++) {
        IUImageSpec spec = {};
        spec.width = depth_width;
        spec.height = depth_height;

        if (iu_read_exr_from_file(exr_files[i], &spec, (void**)
                                  &dev->dummy.depth_images[i]) != SUCCESS)
        {
            xasprintf(err, "Failed to open %s\n", exr_files[i]);
            return false;
        }
        free(exr_files[i]);

        depth_width = spec.width;
        depth_height = spec.height;
    }
    exr_files.clear();

    dev->depth_camera_intrinsics.width = depth_width;
    dev->depth_camera_intrinsics.height = depth_height;
    dev->depth_camera_intrinsics.cx = depth_width / 2;
    dev->depth_camera_intrinsics.cy = depth_height / 2;
    //TODO: fill in .fx and .fy based on vertical_fov in meta.json
    dev->depth_format = GM_FORMAT_Z_F16_M;

    int video_width = 0;
    int video_height = 0;
    for (unsigned i = 0; i < png_files.size(); i++) {
        IUImageSpec spec = {};
        spec.width = video_width;
        spec.height = video_height;
        spec.format = IU_FORMAT_U8;

        if (iu_read_png_from_file(png_files[i], &spec, &dev->dummy.lum_images[i],
                                  NULL, NULL) != SUCCESS)
        {
            xasprintf(err, "Failed to open %s\n", png_files[i]);
            return false;
        }
        free(png_files[i]);

        video_width = spec.width;
        video_height = spec.height;
    }
    png_files.clear();

    dev->video_camera_intrinsics.width = video_width;
    dev->video_camera_intrinsics.height = video_height;
    dev->video_camera_intrinsics.cx = video_width / 2;
    dev->video_camera_intrinsics.cy = video_height / 2;
    //TODO: fill in .fx and .fy based on vertical_fov in meta.json
    dev->video_format = GM_FORMAT_LUMINANCE_U8;

    return true;
}

static void
recording_close(struct gm_device *dev)
{
    /* FIXME */
}

static void *
dummy_io_thread_cb(void *userdata)
{
    struct gm_device *dev = (struct gm_device *)userdata;
    uint64_t frame_time_ns = 1000000000 / 30;

    while (true) {
        do {
            uint64_t time = get_time();
            uint64_t duration = time - dev->dummy.time;
            if (duration < frame_time_ns) {
                uint64_t rem = frame_time_ns - duration;
                usleep(rem / 1000);
            } else {
                dev->dummy.time = time;
                break;
            }
        } while (true);

        /* more or less the same as kinect_depth_frame_cb() */
        if (dev->frame_request_requirements & GM_REQUEST_FRAME_DEPTH) {
            half *depth_image = dev->dummy.depth_images[dev->dummy.frame];

            memcpy(dev->depth_buf_back->base.data, depth_image,
                   dev->depth_buf_back->base.len);

            pthread_mutex_lock(&dev->swap_buffers_lock);

            if (dev->depth_buf_ready)
                gm_buffer_unref(&dev->depth_buf_ready->base);
            dev->depth_buf_ready = dev->depth_buf_back;
            dev->depth_buf_back = mem_pool_acquire_buffer(&dev->depth_buf_pool);
            dev->frame_time = dev->dummy.time;
            dev->frame_ready_requirements |= GM_REQUEST_FRAME_DEPTH;

            pthread_mutex_unlock(&dev->swap_buffers_lock);

            pthread_mutex_lock(&dev->request_requirements_lock);
            if ((dev->frame_request_requirements & dev->frame_ready_requirements) ==
                dev->frame_request_requirements)
            {
                notify_frame_locked(dev);
            }
            pthread_mutex_unlock(&dev->request_requirements_lock);
        }

        /* more or less the same as kinect_rgb_frame_cb() */
        if (dev->frame_request_requirements & GM_REQUEST_FRAME_VIDEO) {
            uint8_t *video_image = dev->dummy.lum_images[dev->dummy.frame];

            memcpy(dev->video_buf_back->base.data, video_image,
                   dev->video_buf_back->base.len);

            pthread_mutex_lock(&dev->swap_buffers_lock);

            if (dev->video_buf_ready)
                gm_buffer_unref(&dev->video_buf_ready->base);
            dev->video_buf_ready = dev->video_buf_back;
            dev->video_buf_back = mem_pool_acquire_buffer(&dev->video_buf_pool);
            dev->frame_ready_requirements |= GM_REQUEST_FRAME_VIDEO;

            pthread_mutex_unlock(&dev->swap_buffers_lock);

            pthread_mutex_lock(&dev->request_requirements_lock);
            if ((dev->frame_request_requirements & dev->frame_ready_requirements) ==
                dev->frame_request_requirements)
            {
                notify_frame_locked(dev);
            }
            pthread_mutex_unlock(&dev->request_requirements_lock);
        }

        dev->dummy.frame = (dev->dummy.frame + 1) % dev->dummy.n_images;
    }

    return NULL;
}

static void
recording_start(struct gm_device *dev)
{
    dev->dummy.frame = 0;
    dev->dummy.time = get_time();
    pthread_create(&dev->dummy.io_thread,
                   NULL,
                   dummy_io_thread_cb,
                   dev);
}

struct gm_device *
gm_device_open(struct gm_logger *log,
               struct gm_device_config *config,
               char **err)
{
    struct gm_device *dev = new gm_device();
    bool status;

    dev->log = log;

    dev->video_buf_pool.dev = dev;
    dev->video_buf_pool.name = "video";
    dev->video_buf_pool.max_size = 5;
    dev->video_buf_pool.alloc_mem = device_video_buf_alloc;
    dev->video_buf_pool.free_mem = device_buffer_free;

    dev->depth_buf_pool.dev = dev;
    dev->depth_buf_pool.name = "depth";
    dev->depth_buf_pool.max_size = INT_MAX;
    dev->depth_buf_pool.alloc_mem = device_depth_buf_alloc;
    dev->depth_buf_pool.free_mem = device_buffer_free;

    dev->frame_pool.dev = dev;
    dev->frame_pool.name = "frame";
    dev->frame_pool.max_size = INT_MAX;
    dev->frame_pool.alloc_mem = device_frame_alloc;
    dev->frame_pool.free_mem = device_frame_free;

    switch (config->type) {
    case GM_DEVICE_KINECT:
#ifdef USE_FREENECT
        status = kinect_open(dev, config, err);
#else
        gm_assert(log, 0, "Kinect support not enabled");
#endif
        break;
    case GM_DEVICE_RECORDING:
        status = recording_open(dev, config, err);
        break;
    }

    if (!status) {
        gm_device_close(dev);
        return NULL;
    }

    return dev;
}

void
gm_device_close(struct gm_device *dev)
{
    switch (dev->type) {
    case GM_DEVICE_KINECT:
#ifdef USE_FREENECT
        kinect_close(dev);
#else
        gm_assert(dev->log, 0, "Kinect support not enabled");
#endif
        break;
    case GM_DEVICE_RECORDING:
        recording_close(dev);
        break;
    }

    delete dev;
}

void
gm_device_set_event_callback(struct gm_device *dev,
                             void (*event_callback)(struct gm_device *dev,
                                                    struct gm_device_event *event,
                                                    void *user_data),
                             void *user_data)
{
    dev->event_callback = event_callback;
    dev->callback_data = user_data;
}

void
gm_device_start(struct gm_device *dev)
{
    switch (dev->type) {
    case GM_DEVICE_KINECT:
#ifdef USE_FREENECT
        kinect_start(dev);
#else
        gm_assert(dev->log, 0, "Kinect support not enabled");
#endif
        break;
    case GM_DEVICE_RECORDING:
        recording_start(dev);
        break;
    }
}

struct gm_intrinsics *
gm_device_get_depth_intrinsics(struct gm_device *dev)
{
    return &dev->depth_camera_intrinsics;
}

struct gm_intrinsics *
gm_device_get_video_intrinsics(struct gm_device *dev)
{
    return &dev->video_camera_intrinsics;
}

struct gm_extrinsics *
gm_device_get_depth_to_video_extrinsics(struct gm_device *dev)
{
    return &dev->depth_to_video_extrinsics;
}

void
gm_device_request_frame(struct gm_device *dev, uint64_t requirements)
{
    pthread_mutex_lock(&dev->request_requirements_lock);
    dev->frame_request_requirements = requirements;
    pthread_mutex_unlock(&dev->request_requirements_lock);
}

#if 0
static void
kinect_update_frame(struct gm_device *dev, struct gm_frame *frame)
{
    frame->depth_format = GM_FORMAT_Z_U16_MM;
}

static void
recording_update_frame(struct gm_device *dev, struct gm_frame *frame)
{
    frame->depth_format = GM_FORMAT_Z_F16_M;
}
#endif

struct gm_frame *
gm_device_get_latest_frame(struct gm_device *dev)
{
    struct gm_device_frame *frame = mem_pool_acquire_frame(&dev->frame_pool);

#if 0
    switch (dev->type) {
    case GM_DEVICE_KINECT:
        kinect_update_frame(dev, frame);
        break;
    case GM_DEVICE_RECORDING:
        recording_update_frame(dev, frame);
        break;
    }
#endif

    pthread_mutex_lock(&dev->swap_buffers_lock);

    if (dev->last_frame)
        gm_frame_unref(dev->last_frame);

    dev->last_frame = &frame->base;

    if (dev->frame_ready_requirements & GM_REQUEST_FRAME_DEPTH) {
        frame->base.depth = &dev->depth_buf_ready->base;
        dev->depth_buf_ready = NULL;
        frame->base.depth_format = dev->depth_format;
        gm_assert(dev->log, frame->base.depth != NULL,
                  "Depth ready flag set but buffer missing");
    }
    if (dev->frame_ready_requirements & GM_REQUEST_FRAME_VIDEO) {
        frame->base.video = &dev->video_buf_ready->base;
        dev->video_buf_ready = NULL;
        frame->base.video_format = dev->video_format;
        gm_assert(dev->log, frame->base.video != NULL,
                  "Video ready flag set but buffer missing");
    } else
        assert(0);

    frame->base.timestamp = dev->frame_time;

    dev->frame_ready_requirements = 0;

    /* Get a ref() for the caller */
    gm_frame_ref(&frame->base);

    pthread_mutex_unlock(&dev->swap_buffers_lock);

    /* We should have one ref for dev->last_frame and return a _ref() to the
     * caller so there's no race between the caller claiming a reference and us
     * possibly dropping our own ref.
     */
    gm_assert(dev->log, frame->base.ref == 2,
              "Spurious ref counting for new frame");
    return &frame->base;
}

struct gm_ui_properties *
gm_device_get_ui_properties(struct gm_device *dev)
{
    return &dev->properties_state;
}
