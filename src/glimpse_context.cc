
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>

#include <pthread.h>

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/timing.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/common/random.h>
#include <pcl/common/generate.h>
#include <pcl/common/common.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#ifndef ANDROID
#include <epoxy/gl.h>
#endif

#define PNG_DEBUG 1
#include <png.h>
#include <setjmp.h>

#include "half.hpp"

#include "xalloc.h"
#include "wrapper_image.h"
#include "infer.h"
#include "loader.h"

#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_assets.h"

#include <android/log.h>

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "ctx"
#define LOGI(...) gm_info(ctx->log, __VA_ARGS__)
#define LOGE(...) gm_error(ctx->log, __VA_ARGS__)

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)


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

enum image_format {
    IMAGE_FORMAT_X8,
    IMAGE_FORMAT_XHALF,
    IMAGE_FORMAT_XFLOAT,
};

struct image
{
    enum image_format format;
    int width;
    int height;
    int byte_stride;

    union {
        uint8_t *data_u8;
        half_float::half *data_half;
        float *data_float;
    };
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
    float depth;
    struct color color;
};

struct gm_tracking
{
    struct gm_context *ctx;

    uint64_t depth_capture_timestamp;
    uint64_t video_capture_timestamp;

    uint8_t *depth_rgb;
    uint8_t *label_map;
    uint8_t *label_map_rgb;
    float *label_probs;
    float *joints;
    GlimpsePointXYZRGBA* label_cloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    uint8_t *face_detect_buf;
    size_t face_detect_buf_width;
    size_t face_detect_buf_height;
};

enum gm_rotation {
  GM_ROTATION_UNKNOWN = -1,
  GM_ROTATION_0 = 0,
  GM_ROTATION_90 = 1,
  GM_ROTATION_180 = 2,
  GM_ROTATION_270 = 3
};

struct gm_context
{
    struct gm_logger *log;

    //struct gm_intrinsics color_camera_intrinsics;
    //struct gm_intrinsics rgbir_camera_intrinsics;
    struct gm_intrinsics depth_camera_intrinsics;
    struct gm_intrinsics video_camera_intrinsics;
    struct gm_intrinsics training_camera_intrinsics;

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

    GLuint point_cloud_program;
    GLuint attrib_cloud_bo;
    size_t attrib_cloud_bo_size;
    GLuint attrib_cloud_pos;
    GLuint attrib_cloud_color;
    GLuint uniform_cloud_mvp;

    std::vector<uint8_t> grey_face_detect_scratch;
    std::vector<glimpse::wrapped_image<unsigned char>> grey_face_detect_wrapped_layers;

    std::atomic<bool> need_new_scaled_frame;
    pthread_mutex_t scaled_frame_cond_mutex;
    pthread_cond_t scaled_frame_available_cond;

    GLuint yuv_frame_scale_program;
    GLuint scale_program;

    enum gm_rotation current_attrib_bo_rotation_ = { GM_ROTATION_UNKNOWN };
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

    pthread_mutex_t skel_track_cond_mutex;
    pthread_cond_t skel_track_cond;

    pthread_mutex_t tracking_swap_mutex;
    bool have_tracking;
    struct gm_tracking *tracking_front;
    struct gm_tracking *tracking_mid;
    struct gm_tracking *tracking_back;

    float *joints_back;
    float *joints_mid;
    float *joints_front;
    bool have_joints;

    GlimpsePointXYZRGBA* debug_label_cloud_back;
    GlimpsePointXYZRGBA* debug_label_cloud_mid;
    GlimpsePointXYZRGBA* debug_label_cloud_front;
    bool have_label_cloud;

    int n_labels;

    JSON_Value *joint_map;
    JIParams *joint_params;
    int n_joints;

    float min_depth;
    float max_depth;

    int n_color_stops;
    float color_stops_range;
    struct color_stop *color_stops;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;

    pthread_mutex_t frame_ready_mutex;
    bool frame_ready;
    pthread_cond_t frame_ready_cond;

    void (*event_callback)(struct gm_context *ctx,
                           struct gm_event *event,
                           void *user_data);

    void *callback_data;
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

static GLuint __attribute__((unused))
load_shader(GLenum type, const char *source, char **err)
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
                xasprintf(err, "Could not compile shader %d:\n%s\n", type, buf);
                free(buf);
            }
            glDeleteShader(shader);
            shader = 0;
        }
    }

    return shader;
}

static GLuint __attribute__((unused))
create_program(const char *vertex_source, const char *fragment_source, char **err)
{
    GLuint vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source, err);
    if (!vertex_shader)
        return 0;

    GLuint fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source, err);
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
                xasprintf(err, "Could not link program:\n%s\n", buf);
                free(buf);
            }
        }
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void
gm_context_detect_faces(struct gm_context *ctx)
{
    struct gm_tracking *tracking = ctx->tracking_back;

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

/* XXX: does more than we should need in the end, but works for early testing
 *
 * we output to a half or full float image that can be easily passed to our
 * current infer() code to infer label probabilities.
 *
 * We also output to another 'organised' xyzrgba point cloud in the layout of
 * an image so it's easy to add the inferred label colors to this but then
 * also convenient to pass the points directly to opengl.
 */
struct image *
reproject_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                      GlimpsePointXYZRGBA *rgba_cloud,
                      const struct gm_intrinsics *intrinsics,
                      enum image_format fmt)
{
    int width = intrinsics->width;
    float half_width = width / 2.0f;
    int height = intrinsics->height;
    float half_height = height / 2.0f;

    assert(fmt == IMAGE_FORMAT_XHALF || fmt == IMAGE_FORMAT_XFLOAT);

    float h_fov = 2.0 * atan(0.5 * width / intrinsics->fx);
    float v_fov = 2.0 * atan(0.5 * height / intrinsics->fy);

    float tan_half_hfov = tan(h_fov / 2.0); // XXX: or just 0.5 * width / intrinsics->fx
    float tan_half_vfov = tan(v_fov / 2.0);

    for (int i = 0; i < width * height; i++) {
        rgba_cloud[i].x = HUGE_VALF;
        rgba_cloud[i].y = HUGE_VALF;
        rgba_cloud[i].z = HUGE_VALF;
        rgba_cloud[i].rgba = 0x00000000;
    }

    /* XXX: curious to measure what impact there might be with using a soft
     * float16 implementation for representing intermediate depth buffers
     * so just naively duplicating the code for now, for profiling...
     */
    struct image *ret = (struct image *)xmalloc(sizeof(*ret));
    if (fmt == IMAGE_FORMAT_XFLOAT) {
        float *img;

        ret->format = IMAGE_FORMAT_XFLOAT;
        ret->width = width;
        ret->height = height;
        ret->byte_stride = sizeof(float) * width;
        ret->data_float = (float *)xmalloc(sizeof(float) * width * height);

        img = ret->data_float;

        /* Our training set had a background depth of 1000 */
        for (int i = 0; i < width * height; i++)
            img[i] = HUGE_DEPTH;

        /* XXX: we don't assume that cloud->width/height are the same as
         * width/height. The pcl point cloud might not be 2d and if it is
         * we might be projecting a high resolution point cloud into a
         * low resolution depth buffer according to our training camera
         * intrinsics
         */
        for (unsigned cy = 0; cy < cloud->height; cy++) {
            pcl::PointXYZ *row = &cloud->points[cy * cloud->width];
            for (unsigned cx = 0; cx < cloud->width; cx++) {
                glm::vec3 point(row[cx].x, row[cx].y, row[cx].z);
                glm::vec2 ndc_point;
                glm::vec2 pos;

                if (isnan(point.x) || isinf(point.x) ||
                    isnan(point.y) || isinf(point.y) ||
                    !isnormal(point.z) || point.z >= HUGE_DEPTH)
                    continue;

                float hfield_width = tan_half_hfov * point.z;
                float vfield_height = -tan_half_vfov * point.z;

                ndc_point.x = point.x / hfield_width;
                ndc_point.y = point.y / vfield_height;

                if (ndc_point.x < -1.0f || ndc_point.x >= 1.0f ||
                    ndc_point.y < -1.0f || ndc_point.y >= 1.0f)
                    continue;

                pos.x = (ndc_point.x + 1.0f) * half_width;
                pos.y = (ndc_point.y + 1.0f) * half_height;
                int x = pos.x;
                int y = pos.y;

                int off = width * y + x;
                img[off] = point.z;
                rgba_cloud[off].x = point.x;
                rgba_cloud[off].y = point.y;
                rgba_cloud[off].z = point.z;
                rgba_cloud[off].rgba = 0xffffffff;
            }
        }
    } else if (fmt == IMAGE_FORMAT_XHALF) {
        half *img;

        ret->format = IMAGE_FORMAT_XHALF;
        ret->width = width;
        ret->height = height;
        ret->byte_stride = sizeof(half) * width;
        ret->data_half = (half *)xmalloc(sizeof(half) * width * height);

        img = ret->data_half;

        /* Our training set had a background depth of 1000 */
        for (int i = 0; i < width * height; i++)
            img[i] = HUGE_DEPTH;

        /* XXX: we don't assume that cloud->width/height are the same as
         * width/height. The pcl point cloud might not be 2d and if it is
         * we might be projecting a high resolution point cloud into a
         * low resolution depth buffer according to our training camera
         * intrinsics
         */
        for (unsigned cy = 0; cy < cloud->height; cy++) {
            pcl::PointXYZ *row = &cloud->points[cy * cloud->width];
            for (unsigned cx = 0; cx < cloud->width; cx++) {
                glm::vec3 point(row[cx].x, row[cx].y, row[cx].z);
                glm::vec2 ndc_point;
                glm::vec2 pos;

                if (isnan(point.x) || isinf(point.x) ||
                    isnan(point.y) || isinf(point.y) ||
                    !isnormal(point.z) || point.z >= HUGE_DEPTH)
                    continue;

                float hfield_width = tan_half_hfov * point.z;
                float vfield_height = -tan_half_vfov * point.z;

                ndc_point.x = point.x / hfield_width;
                ndc_point.y = point.y / vfield_height;

                if (ndc_point.x < -1.0f || ndc_point.x >= 1.0f ||
                    ndc_point.y < -1.0f || ndc_point.y >= 1.0f)
                    continue;

                pos.x = (ndc_point.x + 1.0f) * half_width;
                pos.y = (ndc_point.y + 1.0f) * half_height;
                int x = pos.x;
                int y = pos.y;

                int off = width * y + x;
                img[off] = point.z;
                rgba_cloud[off].x = point.x;
                rgba_cloud[off].y = point.y;
                rgba_cloud[off].z = point.z;
                rgba_cloud[off].rgba = 0xffffffff;
            }
        }
    } else {
        assert(!"reached");
    }

    return ret;
}

void
free_image(struct image *image)
{
    free(image->data_u8);
    free(image);
}

void
gm_context_track_skeleton(struct gm_context *ctx)
{
    struct gm_tracking *tracking = ctx->tracking_back;

    uint64_t start, end, duration;

    // X increases to the right
    // Y increases upwards
    // Z increases outwards

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tracking->cloud;

#if 0
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(cloud);
    crop.setMin(Eigen::Vector4f(-5, -5, ctx->min_depth, 1));
    crop.setMax(Eigen::Vector4f(5, 5, ctx->max_depth, 1));
    crop.filter(*cloud);
#endif

    LOGI("Processing cloud with %d points\n", (int)cloud->points.size());
    start = get_time();

#if 1
    pcl::PassThrough<pcl::PointXYZ> passZ(true);
    passZ.setInputCloud(cloud);
    passZ.setFilterFieldName ("z");
    passZ.setFilterLimits(ctx->min_depth, ctx->max_depth);
    passZ.filter(*cloud);

    LOGI("Cloud has %d points after depth filter\n", (int)cloud->points.size());
#endif

#if 1
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01, 0.01, 0.01);
    vg.filter(*cloud);
    LOGI("Cloud has %d points after voxel grid\n", (int)cloud->points.size());
#endif

#if 1
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_floor(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> passY;
    passY.setInputCloud(cloud);
    passY.setUserFilterValue(HUGE_DEPTH);
    passY.setKeepOrganized(true);
    passY.setFilterFieldName ("y");
    // XXX: Here we're assuming the camera may be on the ground, but ideally
    //      we'd use a height sensor reading here (minus a threshold).
    passY.setFilterLimits(-FLT_MAX, 0.0);
    passY.filter(*cloud_floor);

    assert(cloud_floor->points.size() == cloud->points.size());

    int n_floor_points = cloud->points.size() -
      passY.getRemovedIndices()->size();
    LOGI("Cloud possible floor subset has %d points\n", n_floor_points);
#endif

#if 1
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(false);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setInputCloud(cloud_floor);

    seg.setMaxIterations(250);
    seg.setDistanceThreshold(0.03);

    // XXX: We're assuming that the camera here is perpendicular to the floor
    //      and give a generous threshold, but ideally we'd use device sensors
    //      to detect orientation and use a slightly less broad angle here.
    seg.setAxis(Eigen::Vector3f(0.f, 1.f, 0.f));
    seg.setEpsAngle(M_PI/180.0 * 15.0);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setUserFilterValue(HUGE_DEPTH);
    extract.setKeepOrganized(true);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    int n_removed_points = 0;
    int min_removal_size = n_floor_points * 0.25;
    do {
        seg.segment(*inliers, *coefficients);

        int n_inliers = (int)inliers->indices.size();
        if (n_inliers == 0 || n_inliers < min_removal_size) {
            break;
        }
        n_removed_points += n_inliers;

        LOGI("Removing %d points\n", n_inliers);
#if 0
        if (i == 0) {
            for (uint32_t j = 0; j < inliers->indices.size(); j++) {
                LOGI("Removing (%.2f, %.2f, %.2f)\n",
                     (float)cloud_floor->points[inliers->indices[j]].x,
                     (float)cloud_floor->points[inliers->indices[j]].y,
                     (float)cloud_floor->points[inliers->indices[j]].z);
            }
        }
#endif

        extract.setNegative(true);
        extract.setIndices(inliers);
        extract.filterDirectly(cloud);
        extract.filterDirectly(cloud_floor);
    } while (n_removed_points < n_floor_points * 0.5);

    // Remove HUGE_DEPTH points
    passZ.setInputCloud(cloud);
    passZ.setKeepOrganized(false);
    passZ.setFilterLimits(HUGE_DEPTH, FLT_MAX);
    passZ.setFilterLimitsNegative(true);
    passZ.filter(*cloud);

    LOGI("Removed %d floor candidate points\n", n_removed_points);
#endif

#if 1
    // Use Euclidean cluster extraction to look at only the largest cluster of
    // points (which we hope is the human)

    // Creating the KdTree object for the search method of the extraction
    /*pcl::search::KdTree<pcl::PointXYZ>::Ptr
      tree(new pcl::search::KdTree<pcl::PointXYZ>);*/

    // Seems Octree is faster than KdTree for this use-case. This appears to
    // be highly dependent on the resolution, which I assume should be
    // tweaked alongside the voxel-grid resolution above and the minimum
    // cluster size below.
    pcl::search::Octree<pcl::PointXYZ>::Ptr
      tree(new pcl::search::Octree<pcl::PointXYZ>(0.1));

    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.03);
    ec.setMinClusterSize (cloud->points.size() * 0.1);
    ec.setMaxClusterSize (cloud->points.size());
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    if (cluster_indices.size() > 0) {
        const pcl::PointIndices* largest_cluster = &cluster_indices[0];
        for (uint32_t i = 1; i < cluster_indices.size(); i++) {
            if (cluster_indices[i].indices.size() >
                largest_cluster->indices.size()) {
                largest_cluster = &cluster_indices[i];
            }
        }

        LOGI("Largest cluster size: %d\n", (int)largest_cluster->indices.size());

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator it =
             largest_cluster->indices.begin();
             it != largest_cluster->indices.end (); ++it) {
            cloud_cluster->points.push_back (cloud->points[*it]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        cloud.swap(cloud_cluster);
    }
#endif

    end = get_time();
    duration = end - start;
    LOGI("Human segmentation took %.3f%s",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    if (ctx->depth_camera_intrinsics.width == 0 ||
        ctx->depth_camera_intrinsics.height == 0)
    {
        LOGE("Skipping detection: depth camera intrinsics not initialized\n");
        return;
    } else {
        LOGI("depth intrinsics: w=%d, h=%d\n",
             ctx->depth_camera_intrinsics.width,
             ctx->depth_camera_intrinsics.height);
    }

    start = get_time();
    struct image *depth_img =
        reproject_point_cloud(cloud,
                              tracking->label_cloud,
                              &ctx->training_camera_intrinsics,
                              IMAGE_FORMAT_XHALF);
    int width = depth_img->width;
    int height = depth_img->height;

    if (width == 0 || height == 0) {
        LOGE("Skipping detection: bad re-projected depth image size: %dx%d\n",
             width, height);
        return;
    }

    end = get_time();
    duration = end - start;
    LOGI("People Detector: re-projected point cloud in %.3f%s, w=%d, h=%d\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration),
         (int)width,
         (int)height);


    if (!ctx->decision_trees) {
        LOGE("People detector: exiting: No decision trees loaded");
        return;
    }
    LOGI("People Detector: starting label inference: n_trees=%d, w=%d, h=%d, data=%p\n",
         ctx->n_decision_trees, width, height, depth_img->data_half);
    start = get_time();
    infer_labels(ctx->decision_trees,
                 ctx->n_decision_trees,
                 depth_img->data_half,
                 width,
                 height,
                 tracking->label_probs);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: ran label probability inference in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    uint8_t n_labels = ctx->decision_trees[0]->header.n_labels;

    start = get_time();
    float *weights = calc_pixel_weights(depth_img->data_half,
                                        tracking->label_probs,
                                        width, height,
                                        ctx->n_labels,
                                        ctx->joint_map);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: calculated pixel weights in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

#if 1
    if (ctx->joint_params) {
        start = get_time();
        float vfov =  (2.0f * atanf(0.5 * height /
                                    ctx->training_camera_intrinsics.fy)) *
          180 / M_PI;
        infer_joints_fast(depth_img->data_half,
                          tracking->label_probs,
                          weights,
                          width, height,
                          ctx->n_labels,
                          ctx->joint_map,
                          vfov,
                          ctx->joint_params->joint_params,
                          tracking->joints);
        end = get_time();
        duration = end - start;
        LOGI("People Detector: inferred joints in %.3f%s\n",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));
    }
#endif
    free(weights);
    weights = NULL;

    LOGI("People Detector: colorizing most probable labels. n_labels=%d, data=%p\n",
         n_labels, tracking->label_probs);

    uint8_t *label_map = tracking->label_map;
    uint8_t *rgb_label_map = tracking->label_map_rgb;

    // colorize cloud based on labels for debug
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t label = 0;
            float pr = -1.0;
            int pos = y * width + x;
            float *pr_table = &tracking->label_probs[pos * n_labels];
            for (uint8_t l = 0; l < n_labels; l++) {
                if (pr_table[l] > pr) {
                    label = l;
                    pr = pr_table[l];
                }
            }
            label_map[pos] = label;

            uint8_t r = default_palette[label].red;
            uint8_t g = default_palette[label].green;
            uint8_t b = default_palette[label].blue;

            uint32_t col = r<<24|g<<16|b<<8|0xff;
            tracking->label_cloud[pos].rgba = col;
            //tracking->label_cloud[pos].rgba = 0xffff80ff;

            rgb_label_map[pos * 3] = r;
            rgb_label_map[pos * 3 + 1] = g;
            rgb_label_map[pos * 3 + 2] = b;
        }
    }

    free_image(depth_img);
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
        GM_REQUEST_FRAME_DEPTH | GM_REQUEST_FRAME_LUMINANCE;

    ctx->event_callback(ctx, event, ctx->callback_data);
}

static void
notify_tracking(struct gm_context *ctx)
{
    struct gm_event *event = event_alloc(GM_EVENT_TRACKING_READY);

    ctx->event_callback(ctx, event, ctx->callback_data);
}

static void *
detector_thread_cb(void *data)
{
    struct gm_context *ctx = (struct gm_context *)data;

    LOGE("DetectorRun");

    uint64_t start = get_time();
    ctx->detector = dlib::get_frontal_face_detector();
    uint64_t end = get_time();
    uint64_t duration = end - start;

    LOGE("Initialized Dlib frontal face detector: %.3f%s",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    //LOGI("Dropped all but the first (front-facing HOG) from the DLib face detector");
    //ctx->detector.w.resize(1);

    //LOGI("Detector debug %p", &ctx->detector.scanner);

    char *err = NULL;
    struct gm_asset *predictor_asset =
        gm_asset_open("shape_predictor_68_face_landmarks.dat",
                      GM_ASSET_MODE_BUFFER,
                      &err);
    if (predictor_asset) {
        const void *buf = gm_asset_get_buffer(predictor_asset);
        off_t len = gm_asset_get_length(predictor_asset);
        std::istringstream stream_in(std::string((char *)buf, len));
        try {
            dlib::deserialize(ctx->face_feature_detector, stream_in);
        } catch (dlib::serialization_error &e) {
            LOGI("Failed to deserialize shape predictor: %s", e.info.c_str());
        }

        LOGI("Mapped shape predictor asset %p, len = %d", buf, (int)len);
        gm_asset_close(predictor_asset);
    } else {
        LOGE("Failed to open shape predictor asset: %s", err);
        free(err);
    }

    while(1) {
        LOGI("Waiting for new frame to start tracking\n");
        pthread_mutex_lock(&ctx->frame_ready_mutex);
        while (!ctx->frame_ready) {
            pthread_cond_wait(&ctx->frame_ready_cond, &ctx->frame_ready_mutex);
        }
        ctx->frame_ready = false;
        pthread_mutex_unlock(&ctx->frame_ready_mutex);

        /* While downsampling on the CPU we currently do that synchronosuly
         * when we are notified of a new frame.
         */
#ifdef DOWNSAMPLE_ON_GPU
        LOGI("Waiting for new scaled frame for face detection");
        pthread_mutex_lock(&ctx->scaled_frame_cond_mutex);
        ctx->need_new_scaled_frame = true;
        while (ctx->need_new_scaled_frame)
            pthread_cond_wait(&ctx->scaled_frame_available_cond, &ctx->scaled_frame_cond_mutex);
        pthread_mutex_unlock(&ctx->scaled_frame_cond_mutex);
#endif

        LOGI("Starting tracking iteration\n");

        //gm_context_detect_faces(ctx);

        gm_context_track_skeleton(ctx);
        LOGI("Finished skeletal tracking");

        pthread_mutex_lock(&ctx->tracking_swap_mutex);
        std::swap(ctx->tracking_back, ctx->tracking_mid);
        ctx->have_tracking = true;
        pthread_mutex_unlock(&ctx->tracking_swap_mutex);

        notify_tracking(ctx);

        LOGI("Requesting new frame for skeletal tracking");
        /* We throttle frame aquisition according to our tracking rate... */
        request_frame(ctx);
    }

    pthread_exit((void *)1);

    return NULL;
}

static void
free_tracking(struct gm_tracking *tracking)
{
    free(tracking->label_map);
    free(tracking->label_map_rgb);
    free(tracking->label_probs);
    free(tracking->joints);
    free(tracking->label_cloud);

    free(tracking->depth_rgb);

    free(tracking->face_detect_buf);

    delete tracking;
}

static struct gm_tracking *
alloc_tracking(struct gm_context *ctx)
{
    struct gm_tracking *tracking = new gm_tracking();

    tracking->ctx = ctx;

    int labels_width = ctx->training_camera_intrinsics.width;
    int labels_height = ctx->training_camera_intrinsics.height;

    tracking->label_map = (uint8_t *)xcalloc(labels_width * labels_height, 1);
    tracking->label_map_rgb = (uint8_t *)xcalloc(labels_width * labels_height, 3);
    tracking->label_probs = (float *)xcalloc(labels_width *
                                             labels_height *
                                             sizeof(float) * ctx->n_labels, 1);
    tracking->joints = (float *)xcalloc(ctx->n_joints, 3 * sizeof(float));
    tracking->label_cloud = (GlimpsePointXYZRGBA *)
        xcalloc(labels_width * labels_height, sizeof(GlimpsePointXYZRGBA));

    tracking->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    int depth_width = ctx->depth_camera_intrinsics.width;
    int depth_height = ctx->depth_camera_intrinsics.height;

    assert(depth_width);
    assert(depth_height);

    tracking->depth_rgb = (uint8_t *)xcalloc(depth_width * depth_height, 3);

    int video_width = ctx->video_camera_intrinsics.width;
    int video_height = ctx->video_camera_intrinsics.height;

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

    tracking->face_detect_buf =
        (uint8_t *)xcalloc(tracking->face_detect_buf_width *
                           tracking->face_detect_buf_height, 1);

    return tracking;
}


void
gm_context_destroy(struct gm_context *ctx)
{
    free_tracking(ctx->tracking_front);
    free_tracking(ctx->tracking_mid);
    free_tracking(ctx->tracking_back);

    free(ctx->color_stops);

    for (int i = 0; i < ctx->n_decision_trees; i++)
        free_tree(ctx->decision_trees[i]);
    xfree(ctx->decision_trees);

    free_jip(ctx->joint_params);

    json_value_free(ctx->joint_map);

    delete ctx;
}

static struct color
get_color_from_stops(struct gm_context *ctx, float depth)
{
    struct color_stop *stops = ctx->color_stops;
    int n_stops = ctx->n_color_stops;
    float range = ctx->color_stops_range;

    int i = (int)((fmax(0, fmin(range, depth)) / range) * n_stops) + 1;

    if (i < 1)
        i = 1;
    else if (i >= n_stops)
        i = n_stops - 1;

    float t = (depth - stops[i - 1].depth) /
        (stops[i].depth - stops[i - 1].depth);

    struct color col0 = stops[i - 1].color;
    struct color col1 = stops[i].color;

    float r = (1.0f - t) * col0.r + t * col1.r;
    float g = (1.0f - t) * col0.g + t * col1.g;
    float b = (1.0f - t) * col0.b + t * col1.b;

    return { (uint8_t)r, (uint8_t)g, (uint8_t)b };
}

static struct color
color_from_depth(float depth, float range, int n_stops)
{
    static const uint32_t rainbow[] = {
        0xffff00ff, //yellow
        0x0000ffff, //blue
        0x00ff00ff, //green
        0xff0000ff, //red
        0x00ffffff, //cyan
    };

    int stop = (int)((fmax(0, fmin(range, depth)) / range) * n_stops);
    float f = powf(0.8, floorf(stop / ARRAY_LEN(rainbow)));
    int band = ((int)stop) % ARRAY_LEN(rainbow);

    uint32_t rgba = rainbow[band];

    float r = ((rgba & 0xff000000) >> 24) / 255.0f;
    float g = ((rgba & 0xff0000) >> 16) / 255.0f;
    float b = ((rgba & 0xff00) >> 8) / 255.0f;

    r *= f;
    g *= f;
    b *= f;

    return { (uint8_t)(r * 255.f),
             (uint8_t)(g * 255.f),
             (uint8_t)(b * 255.f) };
}

static void
alloc_rgb_color_stops(struct gm_context *ctx)
{
    int range = 5000;
    float range_m = 5.f;
    int step = 250;
    int n_stops = range / step;

    ctx->n_color_stops = n_stops;
    ctx->color_stops_range = range_m;
    ctx->color_stops = (struct color_stop *)xcalloc(sizeof(struct color_stop), n_stops);

    struct color_stop *stops = ctx->color_stops;

    for (int i = 0; i < n_stops; i++) {
        stops[i].depth = (i * step) / 1000.f;
        stops[i].color = color_from_depth(stops[i].depth, range_m, n_stops);
    }
}

struct gm_context *
gm_context_new(struct gm_logger *logger,
               char **err)
{
    /* NB: we can't just calloc this struct since it contains C++ class members
     * that need to be constructed appropriately
     */
    struct gm_context *ctx = new gm_context();

    ctx->log = logger;

    struct gm_ui_property prop;

    ctx->min_depth = 0.5;
    prop = gm_ui_property();
    prop.name = "min_depth";
    prop.desc = "throw away points nearer than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_ptr = &ctx->min_depth;
    prop.min = 0.0;
    prop.max = 10;
    ctx->properties.push_back(prop);

    ctx->max_depth = 3.5;
    prop = gm_ui_property();
    prop.name = "max_depth";
    prop.desc = "throw away points further than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_ptr = &ctx->max_depth;
    prop.min = 0.5;
    prop.max = 10;
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    pthread_cond_init(&ctx->skel_track_cond, NULL);
    pthread_mutex_init(&ctx->skel_track_cond_mutex, NULL);


#ifdef ANDROID
#error "TODO: call gm_assets_android_set_manager()"
#endif

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
        xsnprintf(json_name, sizeof(json_name), "tree%u.rdt", i);

        char *catch_err = NULL;
        struct gm_asset *tree_asset = gm_asset_open(rdt_name,
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
            tree_asset = gm_asset_open(json_name,
                                       GM_ASSET_MODE_BUFFER,
                                       &open_err);
            if (!tree_asset) {
                xasprintf(err, "Failed to open tree%u.rdt and tree%u.json: %s",
                          i, i, open_err);
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
            LOGI("Failed to load %s", name);
            break;
        }

        ctx->n_decision_trees++;
    }

    if (!ctx->n_decision_trees) {
        xasprintf(err, "Failed to open any decision tree assets");
        gm_context_destroy(ctx);
        return NULL;
    }

    ctx->n_labels = ctx->decision_trees[0]->header.n_labels;

    /* XXX: maybe make it an explicit api to start running detection
     */
    int ret = pthread_create(&ctx->detect_thread,
                             nullptr, /* default attributes */
                             detector_thread_cb,
                             ctx);
    if (ret != 0) {
        xasprintf(err, "Failed to start face detector thread: %s", strerror(ret));
        gm_context_destroy(ctx);
        return NULL;
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
    struct gm_asset *joint_map_asset = gm_asset_open("joint-map.json",
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
            xasprintf(err, "Failed to open joint map\n");
            gm_context_destroy(ctx);
            return NULL;
        }

        ctx->n_joints = json_array_get_count(json_array(ctx->joint_map));
    } else {
        xasprintf(err, "Failed to open joint-map.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }


    ctx->joint_params = NULL;

    struct gm_asset *joint_params_asset =
        gm_asset_open("joint-params.json",
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
            xasprintf(err, "Failed to laod joint params from json");
            gm_context_destroy(ctx);
            return NULL;
        }
    } else {
        xasprintf(err, "Failed to open joint-params.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }

    alloc_rgb_color_stops(ctx);

    return ctx;
}

void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics)
{
    ctx->depth_camera_intrinsics = *intrinsics;
}

void
gm_context_set_video_camera_intrinsics(struct gm_context *ctx,
                                       struct gm_intrinsics *intrinsics)
{
    ctx->video_camera_intrinsics = *intrinsics;
}

void
update_tracking_luminance(struct gm_context *ctx,
                          struct gm_tracking *tracking,
                          enum gm_format format,
                          uint8_t *luminance,
                          uint64_t timestamp)
{
    int width = ctx->video_camera_intrinsics.width;
    int height = ctx->video_camera_intrinsics.height;

    assert(format == GM_FORMAT_LUMINANCE_U8);

    if (!ctx->grey_width) {
        ctx->grey_width = width;
        ctx->grey_height = height;

#ifndef DOWNSAMPLE_ON_GPU
#ifdef DOWNSAMPLE_1_4
        ctx->grey_buffer_1_2.resize((width / 2) * (height / 2));
#endif
#endif
    }

#ifndef DOWNSAMPLE_ON_GPU
    uint64_t start, end, duration_ns;

#ifdef DOWNSAMPLE_1_2
#ifdef DOWNSAMPLE_1_4
    /* 1/4 resolution */
    glimpse::wrapped_image<unsigned char> orig_grey_img;
    orig_grey_img.wrap(ctx->grey_width,
                       ctx->grey_height,
                       ctx->grey_width, //stride
                       static_cast<unsigned char *>(luminance));

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
    glimpse::wrapped_image<unsigned char> orig_grey_img;
    orig_grey_img.wrap(ctx->grey_width,
                       ctx->grey_height,
                       ctx->grey_width, //stride
                       static_cast<unsigned char *>(luminance));

    glimpse::wrapped_image<unsigned char> grey_1_2_img;
    grey_1_2_img.wrap(width / 2,
                      height / 2,
                      width / 2, //stride
                      static_cast<unsigned char *>(tracking->face_detect_buf));
#endif
#else
    /* full resolution */
    memcpy(tracking->face_detect_buf, luminance, width * height);
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

    tracking->video_capture_timestamp = timestamp;
}

static void
update_tracking_depth_from_buffer(struct gm_context *ctx,
                                  struct gm_tracking *tracking,
                                  enum gm_format format,
                                  void *depth,
                                  uint64_t timestamp)
{
    struct gm_intrinsics *intrinsics = &ctx->depth_camera_intrinsics;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tracking->cloud;

    int width = ctx->depth_camera_intrinsics.width;
    int height = ctx->depth_camera_intrinsics.height;

    uint64_t start = get_time();

    cloud->points.clear();
    cloud->width = 0;
    cloud->height = 1;
    cloud->is_dense = true;

    float inv_fx = 1.0f / intrinsics->fx;
    float inv_fy = -1.0f / intrinsics->fy;
    float cx = intrinsics->cx;
    float cy = intrinsics->cy;


    /* With this foreach macro the following block of code will have access to
     * x, y, z, off and depth_m variables. (off = y * width + x)
     */
#define foreach_xy_off(width, height) \
    for (int y = 0; y < height; y++) \
        for (int x = 0, off = 0; x < width && (off = y * width + x); x++)

#define VALIDATE_AND_APPEND_POINT(CLOUD, X, Y, Z) \
    do { \
        if (isnormal(Z) && Z < HUGE_DEPTH) { \
            pcl::PointXYZ point((float)((X - cx) * Z * inv_fx), \
                                (float)((Y - cy) * Z * inv_fy), \
                                Z); \
            CLOUD->points.push_back(point); \
        } \
    } while (0)

#define MAP_DEPTH_TO_RGB(DEPTH_RGB_BUF, OFF, DEPTH) \
    do { \
        struct color rgb = get_color_from_stops(ctx, DEPTH); \
        uint8_t *depth_rgb = DEPTH_RGB_BUF + OFF * 3; \
        depth_rgb[0] = rgb.r; \
        depth_rgb[1] = rgb.g; \
        depth_rgb[2] = rgb.b; \
    } while (0)

    uint8_t *depth_rgb_back = tracking->depth_rgb;

    switch (format) {
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_LUMINANCE_U8:
        assert(0);
        break;
    case GM_FORMAT_Z_U16_MM:
        foreach_xy_off(width, height) {
            float depth_m = ((uint16_t *)depth)[off] / 1000.f;
            VALIDATE_AND_APPEND_POINT(cloud, x, y, depth_m);
            MAP_DEPTH_TO_RGB(depth_rgb_back, off, depth_m);
        }
        break;
    case GM_FORMAT_Z_F32_M:
        foreach_xy_off(width, height) {
            float depth_m = ((float *)depth)[off];
            VALIDATE_AND_APPEND_POINT(cloud, x, y, depth_m);
            MAP_DEPTH_TO_RGB(depth_rgb_back, off, depth_m);
        }
        break;
    case GM_FORMAT_Z_F16_M:
        foreach_xy_off(width, height) {
            float depth_m = ((half *)depth)[off];
            VALIDATE_AND_APPEND_POINT(cloud, x, y, depth_m);
            MAP_DEPTH_TO_RGB(depth_rgb_back, off, depth_m);
        }
        break;
    }

#undef MAP_DEPTH_TO_RGB
#undef VALIDATE_AND_APPEND_POINT
#undef foreach_xy_off

    cloud->width = cloud->points.size();

    uint64_t end = get_time();
    uint64_t duration = end - start;

    LOGI("gm_context_update_depth: projected depth buffer into pcl cloud in %.3f%s",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    tracking->depth_capture_timestamp = timestamp;
}

const uint8_t *
gm_tracking_get_rgb_depth(struct gm_tracking *tracking)
{
    //struct gm_context *ctx = tracking->ctx;

    return tracking->depth_rgb;
}

const GlimpsePointXYZRGBA *
gm_tracking_get_rgb_label_cloud(struct gm_tracking *tracking,
                                int *n_points)
{
    if (n_points)
      *n_points = tracking->ctx->training_camera_intrinsics.width *
                  tracking->ctx->training_camera_intrinsics.height;
    return tracking->label_cloud;
}

const float *
gm_tracking_get_joint_positions(struct gm_tracking *tracking,
                                int *n_joints)
{
    if (n_joints) *n_joints = tracking->ctx->n_joints;
    return tracking->joints;
}

const uint8_t *
gm_tracking_get_rgb_label_map(struct gm_tracking *tracking,
                              int *width,
                              int *height)
{
    struct gm_context *ctx = tracking->ctx;

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return tracking->label_map_rgb;
}

const uint8_t *
gm_tracking_get_label_map(struct gm_tracking *tracking,
                          int *width,
                          int *height)
{
    struct gm_context *ctx = tracking->ctx;

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return tracking->label_map;
}

const float *
gm_tracking_get_label_probabilities(struct gm_tracking *tracking,
                                    int *width,
                                    int *height)
{
    struct gm_context *ctx = tracking->ctx;

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return tracking->label_probs;
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

    update_tracking_depth_from_buffer(ctx,
                                      ctx->tracking_back,
                                      frame->depth_format,
                                      frame->depth,
                                      frame->timestamp);

    assert(frame->video_format == GM_FORMAT_LUMINANCE_U8);
    update_tracking_luminance(ctx,
                              ctx->tracking_back,
                              frame->video_format,
                              (uint8_t *)frame->video,
                              frame->timestamp);

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    ctx->frame_ready = true;
    pthread_cond_signal(&ctx->frame_ready_cond);
    pthread_mutex_unlock(&ctx->frame_ready_mutex);

    return true;
}

struct gm_tracking *
gm_context_get_latest_tracking(struct gm_context *ctx)
{
    pthread_mutex_lock(&ctx->tracking_swap_mutex);
    std::swap(ctx->tracking_mid, ctx->tracking_front);
    pthread_mutex_unlock(&ctx->tracking_swap_mutex);

    return ctx->tracking_front;
}

void
gm_context_render_thread_hook(struct gm_context *ctx)
{
    /*
     * FIXME: clean all this stuff up...
     */

    if (!ctx->need_new_scaled_frame)
        return;

    /* FIXME: how can we query the info (namely the downsampling rate) at
     * runtime from the ctx->detector.scanner.pyramid_type, instead of hard
     * coding...
     */
    dlib::pyramid_down<6> pyr;

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
    struct gm_tracking *tracking = ctx->tracking_front;

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

        yuv_frame_scale_program_ = create_program(vert_shader, frag_shader, NULL);

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

        scale_program_ = create_program(vert_shader, frag_shader, NULL);

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
    /* Note: we can't allocate these up front in gm_cotext_new because we need
     * to wait until we've been told depth + video camera intrinsics
     */
    ctx->tracking_front = alloc_tracking(ctx);
    ctx->tracking_mid = alloc_tracking(ctx);
    ctx->tracking_back = alloc_tracking(ctx);

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

