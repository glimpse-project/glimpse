
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

#include <libfreenect.h>

#include "half.hpp"
#include "xalloc.h"

#include "image_utils.h"

#include "glimpse_device.h"

#define xsnprintf(dest, fmt, ...) do { \
        if (snprintf(dest, sizeof(dest), fmt,  __VA_ARGS__) >= (int)sizeof(dest)) \
            exit(1); \
    } while(0)


using half_float::half;

struct gm_device
{
    enum gm_device_type type;

    union {
        struct {
            int frame;
            uint64_t time;

            int n_images;
            half **depth_images;
            uint8_t **lum_images;

            pthread_t io_thread;
        } dummy;

        struct {
            freenect_context *fctx;
            freenect_device *fdev;

            int ir_brightness;
            float tilt;
            float accel[3];
            float mks_accel[3];
            pthread_t io_thread;
        } kinect;
    };

    TangoCameraIntrinsics video_camera_intrinsics;
    TangoCameraIntrinsics depth_camera_intrinsics;

    void (*frame_callback)(struct gm_device *dev,
                           struct gm_frame *frame,
                           void *user_data);
    void *frame_callback_data;

    /* What data is required for the next frame?
     * E.g. _DEPETH | _LUMINANCE | _COLOR
     */
    uint64_t frame_request_requirements;

    pthread_mutex_t swap_buffers_lock;

    enum gm_format depth_format;
    void *depth_back;
    void *depth_mid;
    void *depth_front;

    enum gm_format video_format;
    void *video_back;
    void *video_mid;
    void *video_front;

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

static void
kinect_depth_frame_cb(freenect_device *fdev, void *depth, uint32_t timestamp)
{
    struct gm_device *dev = (struct gm_device *)freenect_get_user(fdev);

    if (!(dev->frame_request_requirements & GM_REQUEST_FRAME_DEPTH))
        return;

    pthread_mutex_lock(&dev->swap_buffers_lock);
    std::swap(dev->depth_mid, dev->depth_back);
    dev->frame_ready_requirements |= GM_REQUEST_FRAME_DEPTH;
    freenect_set_depth_buffer(fdev, dev->depth_back);
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
kinect_rgb_frame_cb(freenect_device *fdev, void *yuv, uint32_t timestamp)
{
    struct gm_device *dev = (struct gm_device *)freenect_get_user(fdev);

    if (!(dev->frame_request_requirements & GM_REQUEST_FRAME_LUMINANCE))
        return;

    int width = dev->video_camera_intrinsics.width;
    int height = dev->video_camera_intrinsics.height;

    uint8_t *lum_back = (uint8_t *)dev->video_back;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int in_pos = y * width * 2 + x * 2;
            int out_pos = y * width + x;

            uint8_t lum = ((uint8_t *)yuv)[in_pos + 1];
            lum_back[out_pos] = lum;
        }
    }

    pthread_mutex_lock(&dev->swap_buffers_lock);
    std::swap(dev->video_mid, dev->video_back);
    dev->frame_ready_requirements |= GM_REQUEST_FRAME_LUMINANCE;
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

    dev->kinect.tilt = freenect_get_tilt_degs(tilt_state);

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

    /* FIXME: we can't query intrinsics from libfreenect */
    dev->video_camera_intrinsics = dev->depth_camera_intrinsics;
    dev->video_format = GM_FORMAT_LUMINANCE_U8;

    /* Allocated large enough got _U16_MM data */
    int depth_width = dev->depth_camera_intrinsics.width;
    int depth_height = dev->depth_camera_intrinsics.height;
    dev->depth_front = xmalloc(depth_width * depth_height * 2);
    dev->depth_mid = xmalloc(depth_width * depth_height * 2);
    dev->depth_back = xmalloc(depth_width * depth_height * 2);

    /* allocated large enough for _LUMINANCE format */
    int video_width = dev->video_camera_intrinsics.width;
    int video_height = dev->video_camera_intrinsics.height;
    dev->video_front = xmalloc(video_width * video_height);
    dev->video_mid = xmalloc(video_width * video_height);
    dev->video_back = xmalloc(video_width * video_height);

    freenect_set_video_callback(dev->kinect.fdev, kinect_rgb_frame_cb);
    freenect_set_video_mode(dev->kinect.fdev,
                            freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM,
                                                     FREENECT_VIDEO_YUV_RAW));
    /* XXX: we don't explicitly set a back buffer for video and rely on
     * libfreenect automatically allocating one for us. We are only keeping the
     * luminance which we copy out immediately when notified of a new frame in
     * kinect_rgb_frame_cb which means the mid and front buffers have a
     * different size to the back buffer.
     */

    freenect_set_depth_callback(dev->kinect.fdev, kinect_depth_frame_cb);
    //freenect_set_depth_mode(dev->kinect.fdev,
    //                        freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,
    //                        FREENECT_DEPTH_REGISTERED)); // MM, aligned to RGB
    freenect_set_depth_mode(dev->kinect.fdev,
                            freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,
                                                     FREENECT_DEPTH_MM));
    freenect_set_depth_buffer(dev->kinect.fdev, dev->depth_back);


    struct gm_ui_property prop;

    /* FIXME: need an explicit setter function so we can call
     * freenect_set_ir_brightness()
     */
    prop = gm_ui_property();
    prop.name = "ir_brightness";
    prop.desc = "IR Brightness";
    prop.type = GM_PROPERTY_INT;
    prop.int_ptr = &dev->kinect.ir_brightness;
    prop.min = 0;
    prop.max = 50;
    dev->properties.push_back(prop);

    /* FIXME: need an explicit setter function so we can call
     * freenect_set_tilt_degs()
     */
    prop = gm_ui_property();
    prop.name = "tilt";
    prop.desc = "Tilt";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_ptr = &dev->kinect.tilt;
    prop.min = -30;
    prop.max = 30;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.name = "accel";
    prop.desc = "Accel";
    prop.type = GM_PROPERTY_FLOAT_VEC3;
    prop.float_vec3 = dev->kinect.accel;
    prop.read_only = true;
    dev->properties.push_back(prop);

    prop = gm_ui_property();
    prop.name = "mks_accel";
    prop.desc = "MKS Accel";
    prop.type = GM_PROPERTY_FLOAT_VEC3;
    prop.float_vec3 = dev->kinect.mks_accel;
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

    free(dev->video_front);
    free(dev->video_mid);
    free(dev->video_back);

    free(dev->depth_front);
    free(dev->depth_mid);
    free(dev->depth_back);
}

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

        xsnprintf(next_path, "%s/%s", path, entry->d_name);

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

            dev->kinect.tilt = freenect_get_tilt_degs(state);
            dev->kinect.ir_brightness = freenect_get_ir_brightness(dev->kinect.fdev);

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

        half *depth_image = dev->dummy.depth_images[dev->dummy.frame];
        uint8_t *luminance_image = dev->dummy.lum_images[dev->dummy.frame];

        if (dev->frame_request_requirements & GM_REQUEST_FRAME_LUMINANCE) {
            int video_width = dev->video_camera_intrinsics.width;
            int video_height = dev->video_camera_intrinsics.height;

            memcpy(dev->video_back, luminance_image,
                   video_width * video_height);
        }

        if (dev->frame_request_requirements & GM_REQUEST_FRAME_DEPTH) {
            int depth_width = dev->depth_camera_intrinsics.width;
            int depth_height = dev->depth_camera_intrinsics.height;

            memcpy(dev->depth_back, depth_image,
                   depth_width * depth_height);
        }

        pthread_mutex_lock(&dev->request_requirements_lock);
        if ((dev->frame_request_requirements & dev->frame_ready_requirements) ==
            dev->frame_request_requirements)
        {
            notify_frame_locked(dev);
        }
        pthread_mutex_unlock(&dev->request_requirements_lock);

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
gm_device_open(struct gm_device_config *config, char **err)
{
    struct gm_device *dev = new gm_device();
    bool status;

    switch (config->type) {
    case GM_DEVICE_KINECT:
        status = kinect_open(dev, config, err);
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
        kinect_close(dev);
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
        kinect_start(dev);
        break;
    case GM_DEVICE_RECORDING:
        recording_start(dev);
        break;
    }
}

TangoCameraIntrinsics *
gm_device_get_depth_intrinsics(struct gm_device *dev)
{
    return &dev->depth_camera_intrinsics;
}

TangoCameraIntrinsics *
gm_device_get_video_intrinsics(struct gm_device *dev)
{
    return &dev->video_camera_intrinsics;
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
    struct gm_frame *frame = (struct gm_frame *)xcalloc(sizeof(*frame), 1);

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

    if (dev->frame_ready_requirements & GM_REQUEST_FRAME_DEPTH) {
        std::swap(dev->depth_front, dev->depth_mid);
        frame->depth = dev->depth_front;
        frame->depth_format = dev->depth_format;
        assert(frame->depth);
    }
    if (dev->frame_ready_requirements & GM_REQUEST_FRAME_LUMINANCE) {
        std::swap(dev->video_front, dev->video_mid);
        frame->video = dev->video_front;
        frame->video_format = dev->video_format;
        assert(frame->video);
    } else
        assert(0);

    dev->frame_ready_requirements = 0;

    pthread_mutex_unlock(&dev->swap_buffers_lock);

    return frame;
}

void
gm_device_free_frame(struct gm_device *dev,
                     struct gm_frame *frame)
{
    free(frame);
}

struct gm_ui_properties *
gm_device_get_ui_properties(struct gm_device *dev)
{
    return &dev->properties_state;
}
