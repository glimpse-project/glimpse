
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

#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>

#include <tango_support_api.h>

#include <tango_3d_reconstruction_api.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/timing.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/common/random.h>
#include <pcl/common/generate.h>
#include <pcl/common/common.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#ifndef ANDROID
#include <epoxy/gl.h>
#endif

#define PNG_DEBUG 1
#include <png.h>
#include <setjmp.h>

#include "glimpse_context.h"

#include "plane_fitting.h"

#include "wrapper_image.h"
#include "infer.h"
#include "loader.h"
#include "half.hpp"

#include <android/log.h>
#define LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "glimpse_context", __VA_ARGS__)
#define LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "glimpse_context", __VA_ARGS__)


using half_float::half;
using namespace pcl::common;


#define DOWNSAMPLE_1_2

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

struct gm_context
{
    //TangoCameraIntrinsics color_camera_intrinsics;
    //TangoCameraIntrinsics rgbir_camera_intrinsics;
    TangoCameraIntrinsics depth_camera_intrinsics;
    TangoCameraIntrinsics training_camera_intrinsics;

    AAssetManager *asset_manager;

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
    std::vector<uint8_t> grey_buffer_1_4; //quater size

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
    uint8_t *detect_buf_data;
    size_t detect_buf_width;
    size_t detect_buf_height;


    /* We have slightly awkward producer consumer relationships at the
     * moment, camera frames from Tango service, scaling done via GLES on
     * render thread, face detection done in dedicated thread.
     *
     * For now we use a synchronous pull model.
     *
     * Setting need_new_luminance_cam_frame_ lets the render thread request
     * a new camera frame from Tango.
     *
     * Setting need_new_scaled_frame_ lets the face detect thread request
     * the render thread to build a pyramid of downsampled camera frame
     * images.
     */
    std::atomic<bool> need_new_luminance_cam_frame;
    pthread_mutex_t luminance_cond_mutex;
    pthread_cond_t luminance_available_cond;
    double luminance_timestamp;

    std::atomic<bool> need_new_scaled_frame;
    pthread_mutex_t scaled_frame_cond_mutex;
    pthread_cond_t scaled_frame_available_cond;

    GLuint yuv_frame_scale_program;
    GLuint scale_program;

    TangoSupportRotation current_attrib_bo_rotation_ = { ROTATION_IGNORED };
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

    pthread_mutex_t cloud_swap_mutex;
    /* derived during gm_context_update_depth_from_... */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_back;
    /* ready to be used for skeleton tracking */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pending;
    bool got_cloud; /* pending cloud is valid */
    /* in use by skeleton tracking code */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_processing;

    pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr debug_label_cloud;


    pthread_mutex_t labels_swap_mutex;
    uint8_t *label_map_back;
    uint8_t *label_map_mid;
    uint8_t *label_map_front;
    bool have_label_map;

    uint8_t *label_map_rgb_back;
    uint8_t *label_map_rgb_mid;
    uint8_t *label_map_rgb_front;
    bool have_rgb_label_map;

    float *label_probs_back;
    float *label_probs_mid;
    float *label_probs_front;
    bool have_label_probs;

    int n_labels;

    JSON_Value *joint_map;
    JIParams *joint_params;

    float min_depth;
    float max_depth;

    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;
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


uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

char *
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

float
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
    if (!ctx->detect_buf_data) {
        LOGI("NULL ctx->detect_buf_data");
        return;
    }

    uint64_t start, end, duration_ns;
    std::vector<dlib::rectangle> face_rects(0);
    glimpse::wrapped_image<unsigned char> grey_img;
    dlib::rectangle buf_rect(ctx->detect_buf_width, ctx->detect_buf_height);

    LOGI("New camera frame to process");

    if (ctx->last_faces.size()) {

        LOGI("Searching %d region[s] for faces", (int)ctx->last_faces.size());

        for (dlib::rectangle &rect : ctx->last_faces) {

            rect = dlib::grow_rect(static_cast<dlib::rectangle&>(rect), (long)((float)rect.width() * 0.4f));
            rect.intersect(buf_rect);

            grey_img.wrap(rect.width(),
                          rect.height(),
                          ctx->detect_buf_width, //stride
                          static_cast<unsigned char *>(ctx->detect_buf_data +
                                                       rect.top() * ctx->detect_buf_width +
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
    grey_img.wrap(ctx->detect_buf_width,
                  ctx->detect_buf_height,
                  ctx->detect_buf_width, //stride
                  static_cast<unsigned char *>(ctx->detect_buf_data));

    /* Fall back to checking full frame if the number of detected
     * faces has changed
     */
    if (face_rects.size() != ctx->last_faces.size() ||
        face_rects.size() == 0)
    {
        LOGI("Starting face detection with %dx%d image",
             (int)ctx->detect_buf_width, (int)ctx->detect_buf_height);
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
        landmarks[i].x = (landmarks[i].x / (float)ctx->detect_buf_width) * 2.f - 1.f;
        landmarks[i].y = (landmarks[i].y / (float)ctx->detect_buf_height) * -2.f + 1.f;
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
        grey_debug_buffer_.resize(ctx->detect_buf_width * ctx->detect_buf_height);
        memcpy(&grey_debug_buffer_[0], ctx->detect_buf_data, grey_debug_buffer_.size());
        grey_debug_width_ = ctx->detect_buf_width;
        grey_debug_height_ = ctx->detect_buf_height;
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
                      const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rgba_cloud,
                      const TangoCameraIntrinsics *intrinsics,
                      enum image_format fmt)
{
    int width = intrinsics->width;
    float half_width = width / 2.0;
    int height = intrinsics->height;
    float half_height = height / 2.0;

    assert(fmt == IMAGE_FORMAT_XHALF || fmt == IMAGE_FORMAT_XFLOAT);

    float h_fov = 2.0 * atan(0.5 * width / intrinsics->fx);
    float v_fov = 2.0 * atan(0.5 * height / intrinsics->fy);

    float tan_half_hfov = tan(h_fov / 2.0); // XXX: or just 0.5 * width / intrinsics->fx
    float tan_half_vfov = tan(v_fov / 2.0);

    rgba_cloud->width = width;
    rgba_cloud->height = height;
    rgba_cloud->reserve(width * height);

    for (unsigned i = 0; i < rgba_cloud->size(); i++) {
        rgba_cloud->points[i].x = HUGE_VALF;
        rgba_cloud->points[i].y = HUGE_VALF;
        rgba_cloud->points[i].z = HUGE_VALF;
        rgba_cloud->points[i].rgba = 0x00000000;
    }
    rgba_cloud->is_dense = false;
    LOGI("resized rgba cloud: w=%d, h=%d, size=%d\n",
         rgba_cloud->width,
         rgba_cloud->height,
         (int)rgba_cloud->size());

    /* XXX: curious to measure what impact there might be with using a soft
     * float16 implementation for representing intermediate depth buffers
     * so just naively duplicating the code for now, for profiling...
     */
    if (fmt == IMAGE_FORMAT_XFLOAT) {
        struct image *ret = (struct image *)xmalloc(sizeof(*ret));
        float *img;

        ret->format = IMAGE_FORMAT_XFLOAT;
        ret->width = width;
        ret->height = height;
        ret->byte_stride = sizeof(float) * width;
        ret->data_float = (float *)xmalloc(sizeof(float) * width * height);

        img = ret->data_float;

        /* Our training set had a background depth of 1000 */
        for (int i = 0; i < width * height; i++)
            img[i] = 1000.0;

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

                if (isnan(point.x) || isnan(point.y) || isnan(point.z) || point.z == 0)
                    continue;

                float hfield_width = tan_half_hfov * point.z;
                float vfield_height = tan_half_vfov * point.z;

                ndc_point.x = point.x / hfield_width;
                ndc_point.y = point.y / vfield_height;

                if (ndc_point.x < -1.0f || ndc_point.x > 1.0 ||
                    ndc_point.y < -1.0f || ndc_point.y > 1.0f)
                    continue;

                pos.x = (ndc_point.x + 1.0f) * half_width;
                pos.y = (ndc_point.y + 1.0f) * half_height;
                int x = pos.x + 0.5f; // don't floor - round to nearest int
                int y = pos.y + 0.5f; // don't floor - round to nearest int

                int off = width * y + x;
                img[off] = point.z;
                rgba_cloud->points[off].x = point.x;
                rgba_cloud->points[off].y = point.y;
                rgba_cloud->points[off].z = point.z;
                rgba_cloud->points[off].rgba = 0xffffffff;
            }
        }

        return ret;
    } else if (fmt == IMAGE_FORMAT_XHALF) {
        struct image *ret = (struct image *)xmalloc(sizeof(*ret));
        half *img;

        ret->format = IMAGE_FORMAT_XHALF;
        ret->width = width;
        ret->height = height;
        ret->byte_stride = sizeof(half) * width;
        ret->data_half = (half *)xmalloc(sizeof(half) * width * height);

        img = ret->data_half;

        /* Our training set had a background depth of 1000 */
        for (int i = 0; i < width * height; i++)
            img[i] = 1000.0;

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

                if (isnan(point.x) || isnan(point.y) || isnan(point.z) || point.z == 0)
                    continue;

                float hfield_width = tan_half_hfov * point.z;
                float vfield_height = tan_half_vfov * point.z;

                ndc_point.x = point.x / hfield_width;
                ndc_point.y = point.y / vfield_height;

                if (ndc_point.x < -1.0f || ndc_point.x > 1.0 ||
                    ndc_point.y < -1.0f || ndc_point.y > 1.0f)
                    continue;

                pos.x = (ndc_point.x + 1.0f) * half_width;
                pos.y = (ndc_point.y + 1.0f) * half_height;
                int x = pos.x + 0.5f; // don't floor - round to nearest int
                int y = pos.y + 0.5f; // don't floor - round to nearest int

                int off = width * y + x;
                img[off] = point.z;
                rgba_cloud->points[off].x = point.x;
                rgba_cloud->points[off].y = point.y;
                rgba_cloud->points[off].z = point.z;
                rgba_cloud->points[off].rgba = 0xffffffff;
            }
        }

        return ret;
    }

    assert(!"reached");
    return NULL;
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
    uint64_t start, end, duration;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = ctx->cloud_processing;

    /* if we want to see each individual plane that's found... */
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);

    /* scratch where */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr label_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);


    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;

#if 0
    int min_remaining = cloud->points.size() * 0.3;
    int n_planes = 0;
    while (cloud->points.size() > min_remaining) {
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            LOGI("No more planes could be detected");
            break;
        }

        n_planes++;

        extract.setIndices(inliers);

#if 0
        extract.setNegative(false);
        extract.filter(*cloud_plane);
#endif

        /* XXX: note that computationally it's a lot more costly to extract
         * the inverse, and I'm sure there's opportunity to optimize here
         * if this is a problem
         */
        extract.setNegative(true);
        extract.filter(*cloud_tmp);

        /* XXX: there's also an api for filtering in-place although notably
         * it doesn't change/reduce the size of the cloud it will just
         * replace filtered-out points with NANs. We could also consider
         * using that API.
         */

        cloud.swap(cloud_tmp);
    }

    end = get_time();
    duration = end - start;
    LOGI("People Detector: pre-processed point cloud in %.3f%s, removed %d planes, %d points remaining\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration),
         n_planes,
         (int)cloud->size());
#endif

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
                              label_cloud,
                              &ctx->training_camera_intrinsics,
                              IMAGE_FORMAT_XHALF);
    int width = depth_img->width;
    int height = depth_img->height;

    if (width == 0 || height == 0) {
        LOGE("Skipping detection: bad re-projected depth image size: %dx%d\n",
             width, height);
        return;
    }

    half *depth_buf = depth_img->data_half;
    float min_depth = ctx->min_depth;
    float max_depth = ctx->max_depth;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pos = y * width + x;
            if (depth_buf[pos] < min_depth)
                depth_buf[pos] = 1000;
            else if (depth_buf[pos] > max_depth)
                depth_buf[pos] = 1000;
        }
    }

    end = get_time();
    duration = end - start;
    LOGI("People Detector: re-projected point cloud in %.3f%s, w=%d, h=%d, n_rgba_points=%d\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration),
         (int)width,
         (int)height,
         (int)label_cloud->size());


    if (!ctx->decision_trees) {
        LOGE("People detector: exiting: No decision trees loaded");
        return;
    }
    LOGI("People Detector: starting label inference: n_trees=%d, w=%d, h=%d, data=%p\n",
         ctx->n_decision_trees, width, height, depth_img->data_half);
    start = get_time();
    float *label_probabilities =
        infer_labels(ctx->decision_trees,
                     ctx->n_decision_trees,
                     depth_img->data_half,
                     width,
                     height);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: ran label probability inference in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    uint8_t n_labels = ctx->decision_trees[0]->header.n_labels;

    /* XXX: could avoid this copy if infer labels could be given a buffer
     * to write into...
     */
    memcpy(ctx->label_probs_back,
           label_probabilities,
           width * height * (int)n_labels * sizeof(float));

    start = get_time();
    float *weights = calc_pixel_weights(depth_img->data_half,
                                        label_probabilities,
                                        width, height,
                                        ctx->n_labels,
                                        ctx->joint_map);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: calculated pixel weights in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

#if 1
    start = get_time();
    float vfov =  2.0f * atanf(0.5 * height / ctx->training_camera_intrinsics.fy);
    float *joints = infer_joints(depth_img->data_half,
                                 label_probabilities,
                                 weights,
                                 width, height,
                                 ctx->n_labels,
                                 ctx->joint_map,
                                 vfov,
                                 ctx->joint_params->joint_params);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: inferred joints in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    free(joints);
    joints = NULL;
#endif
    free(weights);
    weights = NULL;

    LOGI("People Detector: colorizing most probable labels. n_labels=%d, data=%p\n",
         n_labels, label_probabilities);

    uint8_t *label_map = ctx->label_map_back;
    uint8_t *rgb_label_map = ctx->label_map_rgb_back;

    // colorize cloud based on labels for debug
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t label = 0;
            float pr = 0.0;
            int pos = y * width + x;
            float *pr_table = &label_probabilities[pos * n_labels];
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
            label_cloud->points[pos].rgba = col;
            //label_cloud->points[pos].rgba = 0xffff80ff;

            rgb_label_map[pos * 3] = r;
            rgb_label_map[pos * 3 + 1] = g;
            rgb_label_map[pos * 3 + 2] = b;
        }
    }

    free_image(depth_img);
    free(label_probabilities);

    pthread_mutex_lock(&ctx->labels_swap_mutex);
    std::swap(ctx->label_map_rgb_back, ctx->label_map_rgb_mid);
    std::swap(ctx->label_map_back, ctx->label_map_mid);
    std::swap(ctx->label_probs_back, ctx->label_probs_mid);
    ctx->have_label_map = true;
    ctx->have_rgb_label_map = true;
    ctx->have_label_probs = true;
    pthread_mutex_unlock(&ctx->labels_swap_mutex);

    pthread_mutex_lock(&ctx->debug_viz_mutex);

    ctx->debug_cloud = cloud;
    ctx->debug_label_cloud = label_cloud;

    pthread_mutex_unlock(&ctx->debug_viz_mutex);
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

    if (ctx->asset_manager) {
        AAsset *predictor_asset = AAssetManager_open(ctx->asset_manager,
                                                     "shape_predictor_68_face_landmarks.dat",
                                                     AASSET_MODE_BUFFER);
        if (predictor_asset) {
            const void *buf = AAsset_getBuffer(predictor_asset);
            off_t len = AAsset_getLength(predictor_asset);
            std::istringstream stream_in(std::string((char *)buf, len));
            try {
                dlib::deserialize(ctx->face_feature_detector, stream_in);
            } catch (dlib::serialization_error &e) {
                LOGI("Failed to deserialize shape predictor: %s", e.info.c_str());
            }

            LOGI("Mapped shape predictor asset %p, len = %d", buf, (int)len);
            AAsset_close(predictor_asset);
        } else {
            LOGE("Failed to open shape predictor asset");
        }

    } else
        LOGE("No asset manager set before starting face detector thread");

    while(1) {
#if 0
        LOGI("Waiting for new scaled frame for face detection");

        pthread_mutex_lock(&ctx->scaled_frame_cond_mutex);
        ctx->need_new_scaled_frame = true;
        while (ctx->need_new_scaled_frame)
            pthread_cond_wait(&ctx->scaled_frame_available_cond, &ctx->scaled_frame_cond_mutex);
        pthread_mutex_unlock(&ctx->scaled_frame_cond_mutex);

        gm_context_detect_faces(ctx);
#endif

        LOGI("Waiting for point cloud for skeletal tracking");
        pthread_mutex_lock(&ctx->skel_track_cond_mutex);
        while (!ctx->got_cloud)
            pthread_cond_wait(&ctx->skel_track_cond, &ctx->skel_track_cond_mutex);
        pthread_mutex_unlock(&ctx->skel_track_cond_mutex);

        pthread_mutex_lock(&ctx->cloud_swap_mutex);
        std::swap(ctx->cloud_pending, ctx->cloud_processing);
        ctx->got_cloud = false;
        pthread_mutex_unlock(&ctx->cloud_swap_mutex);

        gm_context_track_skeleton(ctx);
    }

    pthread_exit((void *)1);

    return NULL;
}

void
gm_context_destroy(struct gm_context *ctx)
{
    free(ctx->label_map_front);
    free(ctx->label_map_mid);
    free(ctx->label_map_back);

    free(ctx->label_map_rgb_front);
    free(ctx->label_map_rgb_mid);
    free(ctx->label_map_rgb_back);

    free(ctx->label_probs_front);
    free(ctx->label_probs_mid);
    free(ctx->label_probs_back);

    delete ctx;
}

struct gm_context *
gm_context_new(char **err)
{
    /* NB: we can't just calloc this struct since it contains C++ class members
     * that need to be constructed appropriately
     */
    struct gm_context *ctx = new gm_context();

    struct gm_ui_property prop;

    prop = gm_ui_property();
    prop.name = "min_depth";
    prop.desc = "throw away points nearer than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_ptr = &ctx->min_depth;
    prop.min = 0;
    prop.max = 10;
    ctx->properties.push_back(prop);

    prop = gm_ui_property();
    prop.name = "max_depth";
    prop.desc = "throw away points further than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_ptr = &ctx->max_depth;
    prop.min = 0;
    prop.max = 10;
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

    ctx->cloud_back.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ctx->cloud_pending.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ctx->cloud_processing.reset(new pcl::PointCloud<pcl::PointXYZ>);

    ctx->debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ctx->debug_label_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);

    pthread_cond_init(&ctx->skel_track_cond, NULL);
    pthread_mutex_init(&ctx->skel_track_cond_mutex, NULL);


#ifndef ANDROID
    ctx->asset_manager = (AAssetManager *)0x1; // HACK
#endif

    /* Load the decision trees immediately so we know how many labels we're
     * dealing with asap.
     */
    AAsset *tree_asset = AAssetManager_open(ctx->asset_manager,
                                            "tree0.rdt",
                                            AASSET_MODE_BUFFER);
    if (tree_asset) {
        const void *buf = AAsset_getBuffer(tree_asset);
        unsigned len = AAsset_getLength(tree_asset);

        ctx->decision_trees = load_forest((uint8_t **)&buf, &len, 1);
        if (ctx->decision_trees) {
            LOGI("Loaded decision tree\n");
            ctx->n_decision_trees = 1;
        }

        AAsset_close(tree_asset);
    }
    if (!ctx->decision_trees) {
        xasprintf(err, "Failed to open decision tree asset");
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

    ctx->need_new_luminance_cam_frame = true;

    int labels_width = 172;
    int labels_height = 224;
    ctx->training_camera_intrinsics.width = labels_width;
    ctx->training_camera_intrinsics.height = labels_height;
    ctx->training_camera_intrinsics.cx = 86;
    ctx->training_camera_intrinsics.cy = 112;
    ctx->training_camera_intrinsics.fx = 217.461437772;
    ctx->training_camera_intrinsics.fy = 217.461437772;

    ctx->label_map_front = (uint8_t *)xcalloc(labels_width * labels_height, 1);
    ctx->label_map_mid = (uint8_t *)xcalloc(labels_width * labels_height, 1);
    ctx->label_map_back = (uint8_t *)xcalloc(labels_width * labels_height, 1);

    ctx->label_map_rgb_front = (uint8_t *)xcalloc(labels_width * labels_height * 3, 1);
    ctx->label_map_rgb_mid = (uint8_t *)xcalloc(labels_width * labels_height * 3, 1);
    ctx->label_map_rgb_back = (uint8_t *)xcalloc(labels_width * labels_height * 3, 1);

    ctx->label_probs_front = (float *)xcalloc(labels_width *
                                              labels_height *
                                              sizeof(float) * ctx->n_labels, 1);
    ctx->label_probs_mid = (float *)xcalloc(labels_width *
                                            labels_height *
                                            sizeof(float) * ctx->n_labels, 1);
    ctx->label_probs_back = (float *)xcalloc(labels_width *
                                             labels_height *
                                             sizeof(float) * ctx->n_labels, 1);

    ctx->joint_map = NULL;
    AAsset *joint_map_asset = AAssetManager_open(ctx->asset_manager,
                                                 "joint-map.json",
                                                 AASSET_MODE_BUFFER);
    if (joint_map_asset) {
        const void *buf = AAsset_getBuffer(joint_map_asset);
        unsigned len = AAsset_getLength(joint_map_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';

        ctx->joint_map = json_parse_string(js_string);

        free(js_string);
        AAsset_close(joint_map_asset);
    }
    if (!ctx->joint_map) {
        xasprintf(err, "Failed to open joint map\n");
        gm_context_destroy(ctx);
        return NULL;
    }

    ctx->joint_params = NULL;
    AAsset *joint_params_asset = AAssetManager_open(ctx->asset_manager,
                                                    "joint-params.json",
                                                    AASSET_MODE_BUFFER);
    if (joint_params_asset) {
        const void *buf = AAsset_getBuffer(joint_params_asset);
        unsigned len = AAsset_getLength(joint_params_asset);

        /* unfortunately parson doesn't support parsing from a buffer with
         * a given length...
         */
        char *js_string = (char *)xmalloc(len + 1);

        memcpy(js_string, buf, len);
        js_string[len] = '\0';
        JSON_Value *root = json_parse_string(js_string);

        ctx->joint_params = joint_params_from_json(root);

        json_value_free(root);
        free(js_string);
        AAsset_close(joint_params_asset);
    }

    return ctx;
}

void
gm_context_set_depth_camera_intrinsics(struct gm_context *ctx,
                                       TangoCameraIntrinsics *intrinsics)
{
    ctx->depth_camera_intrinsics = *intrinsics;
}

void
gm_context_update_luminance(struct gm_context *ctx,
                            double timestamp,
                            int width, int height,
                            uint8_t *luminance)
{
    /* Note we completely synchronize capturing new frames and running face
     * detection to avoid redundant bandwidth overhead copying frames that we
     * don't use
     */
    if (!ctx->need_new_luminance_cam_frame)
        return;


    if (!ctx->grey_width) {
        ctx->grey_width = width;
        ctx->grey_height = height;
        //uv_buffer_offset_ = ctx->grey_width * ctx->grey_height;
        //yuv_size_ = yuv_width_ * yuv_height_ + yuv_width_ * yuv_height_ / 2;

#ifndef USE_GL_EXT_YUV_TARGET_EXT
        ctx->grey_buffer_1_1.resize(ctx->grey_width * ctx->grey_height);
        ctx->grey_buffer_1_2.resize((ctx->grey_width / 2) * (ctx->grey_height / 2));
        ctx->grey_buffer_1_4.resize((ctx->grey_width / 4) * (ctx->grey_height / 4));
#endif

        //yuv_buffers_[0].resize(yuv_size_);
        //yuv_buffers_[1].resize(yuv_size_);
        //yuv_buffers_[2].resize(yuv_size_);
        //LOGE("Allocated yuv_buffers_[]");
    } //else
    //LOGE("Already allocated yuv_buffers_[]");

    uint64_t start, end, duration_ns;

#ifndef USE_GL_EXT_YUV_TARGET_EXT
    start = get_time();
    memcpy(ctx->grey_buffer_1_1.data(), luminance, ctx->grey_buffer_1_1.size());
    end = get_time();
    duration_ns = end - start;
    LOGI("Copying original 1:1 frame too %.3f%s",
         get_duration_ns_print_scale(duration_ns),
         get_duration_ns_print_scale_suffix(duration_ns));
#endif

#ifndef DOWNSAMPLE_ON_GPU

#ifdef DOWNSAMPLE_1_2
    /* We're just taking the luminance and ignoring the chroma for face
     * detection...
     */
    glimpse::wrapped_image<unsigned char> orig_grey_img;
    orig_grey_img.wrap(ctx->grey_width,
                       ctx->grey_height,
                       ctx->grey_width, //stride
                       static_cast<unsigned char *>(luminance));

    glimpse::wrapped_image<unsigned char> grey_1_2_img;
    grey_1_2_img.wrap(ctx->grey_width / 2,
                      ctx->grey_height / 2,
                      ctx->grey_width / 2, //stride
                      static_cast<unsigned char *>(ctx->grey_buffer_1_2.data()));
#endif

#ifdef DOWNSAMPLE_1_4
    glimpse::wrapped_image<unsigned char> grey_1_4_img;
    grey_1_4_img.wrap(ctx->grey_width / 4,
                      ctx->grey_height / 4,
                      ctx->grey_width / 4, //stride
                      static_cast<unsigned char *>(ctx->grey_buffer_1_4.data()));
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

    ctx->detect_buf_data = ctx->grey_buffer_1_4.data();
    ctx->detect_buf_width = ctx->grey_width / 4;
    ctx->detect_buf_height = ctx->grey_height / 4;
#else
    ctx->detect_buf_data = ctx->grey_buffer_1_2.data();
    ctx->detect_buf_width = ctx->grey_width / 2;
    ctx->detect_buf_height = ctx->grey_height / 2;
#endif // DOWNSAMPLE_1_4

#else
    ctx->detect_buf_data = ctx->grey_buffer_1_1.data();
    ctx->detect_buf_width = ctx->grey_width;
    ctx->detect_buf_height = ctx->grey_height;
#endif


#endif // !DOWNSAMPLE_ON_GPU

    ctx->luminance_timestamp = timestamp;

    ctx->need_new_luminance_cam_frame = false;
    pthread_cond_signal(&ctx->luminance_available_cond);

    //pthread_mutex_lock(&yuv_data_mutex_);
    //std::swap(current_copy_buf_, current_ready_buf_);
    //pthread_mutex_unlock(&yuv_data_mutex_);

    /* Note: from observation of the buffer->data pointers I was hoping to get a
     * clue about whether Tango was internally managing a chain of recycled
     * buffers so we could look at avoiding copying here. Although it looks like
     * it does recycle allocations there was a surprisingly large number (13)
     * which made me doubt whether they might actually just be allocating on
     * demand and so 13 is just a quirk of recycling addresses within their
     * allocator. 13 buffers would equate to ~51MB (1920x1080 NV21) which isn't
     * so large as to rule out that Tango might really be holding on to that
     * many buffers internally.
     */
    //LOGE("DEBUG: New color frame available. width = %d, height = %d, format = %s, ptr=%p",
    //     width, height, fmt, buffer->data);
}

void
gm_context_update_depth_from_u16_mm(struct gm_context *ctx,
                                    double timestamp,
                                    int width, int height,
                                    uint16_t *depth_mm)
{
    TangoCameraIntrinsics *intrinsics = &ctx->depth_camera_intrinsics;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = ctx->cloud_back;

    uint64_t start = get_time();

    cloud->width = width;
    cloud->height = height;
    cloud->points.resize(width * height);

    float inv_fx = 1.0f / intrinsics->fx;
    float inv_fy = 1.0f / intrinsics->fy;
    float cx = intrinsics->cx;
    float cy = intrinsics->cy;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int off = y * width + x;
            float depth_m = depth_mm[off] / 1000.0f;

            cloud->points[off].x = (float)((x - cx) * depth_m * inv_fx);
            cloud->points[off].y = (float)((y - cy) * depth_m * inv_fy);
            cloud->points[off].z = depth_m;
        }
    }

    uint64_t end = get_time();
    uint64_t duration = end - start;

    LOGI("gm_context_update_depth_from_u16_mm: projected depth buffer into pcl cloud in %.3f%s",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    pthread_mutex_lock(&ctx->cloud_swap_mutex);
    std::swap(ctx->cloud_back, ctx->cloud_pending);
    ctx->got_cloud = true;
    pthread_cond_signal(&ctx->skel_track_cond);
    pthread_mutex_unlock(&ctx->cloud_swap_mutex);
}

void
gm_context_update_depth_from_tango_cloud(struct gm_context *ctx,
                                         double timestamp,
                                         int width, int height,
                                         TangoPointCloud *point_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = ctx->cloud_pending;

    uint64_t start = get_time();

    cloud->width  = point_cloud->num_points;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    /* Convert TangoPointCloud into libpcl PointCloud
     * XXX: Note libpcl doesn't appear to support a zero-copy way of
     * constructing a PointCloud from existing data :(
     */
    float *tg_points = (float *)point_cloud->points;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = tg_points[i*4 + 0];
        cloud->points[i].y = tg_points[i*4 + 1];
        cloud->points[i].z = tg_points[i*4 + 2];
    }

    uint64_t end = get_time();
    uint64_t duration = end - start;
    LOGI("People Detector: converted Tango cloud into pcl cloud in %.3f%s",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    pthread_mutex_lock(&ctx->cloud_swap_mutex);
    std::swap(ctx->cloud_back, ctx->cloud_pending);
    ctx->got_cloud = true;
    pthread_cond_signal(&ctx->skel_track_cond);
    pthread_mutex_unlock(&ctx->cloud_swap_mutex);
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

    pthread_mutex_lock(&ctx->luminance_cond_mutex);

    LOGI("Waiting for new camera frame to downsample");
    ctx->need_new_luminance_cam_frame = true;
    while (ctx->need_new_luminance_cam_frame)
        pthread_cond_wait(&ctx->luminance_available_cond, &ctx->luminance_cond_mutex);
    pthread_mutex_unlock(&ctx->luminance_cond_mutex);


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

        TangoSupport_getVideoOverlayUVBasedOnDisplayRotation(
            coords, display_rotation_, out_coords);

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
        case ROTATION_0:
            need_portrait_downsample_fb = true;
            LOGI("> rotation = 0");
            break;
        case ROTATION_90:
            need_portrait_downsample_fb = false;
            LOGI("> rotation = 90");
            break;
        case ROTATION_180:
            need_portrait_downsample_fb = true;
            LOGI("> rotation = 180");
            break;
        case ROTATION_270:
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

        ctx->detect_buf_width = rotated_frame_width / 4;
        ctx->detect_buf_height = rotated_frame_height / 4;

        /* TODO: avoid copying out of the PBO later (assuming we can get a
         * cached mapping)
         */
        LOGI("face detect scratch width = %d, height = %d",
             (int)ctx->detect_buf_width,
             (int)ctx->detect_buf_height);
        ctx->grey_face_detect_scratch.resize(ctx->detect_buf_width * ctx->detect_buf_height);
        memcpy(ctx->grey_face_detect_scratch.data(), pbo_ptr, ctx->grey_face_detect_scratch.size());

        ctx->detect_buf_data = ctx->grey_face_detect_scratch.data();
        LOGI("ctx->detect_buf_data = %p", ctx->detect_buf_data);
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    //glDeleteTextures(ctx->pyramid_tex_layers.size(), &ctx->pyramid_tex_layers[0]);
    //glDeleteFramebuffers(pyramid_fbos_.size(), &pyramid_fbos_[0]);

#endif /* DOWNSAMPLE_ON_GPU */

    ctx->need_new_scaled_frame = false;
    pthread_cond_signal(&ctx->scaled_frame_available_cond);
}

const float *
gm_context_get_latest_label_probabilities(struct gm_context *ctx,
                                          int *width,
                                          int *height)
{
    pthread_mutex_lock(&ctx->labels_swap_mutex);
    if (ctx->have_label_probs) {
        std::swap(ctx->label_probs_mid, ctx->label_probs_front);
        ctx->have_label_probs = false;
    }
    pthread_mutex_unlock(&ctx->labels_swap_mutex);

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return ctx->label_probs_front;
}

const uint8_t *
gm_context_get_latest_label_map(struct gm_context *ctx,
                                int *width,
                                int *height)
{
    pthread_mutex_lock(&ctx->labels_swap_mutex);
    if (ctx->have_label_map) {
        std::swap(ctx->label_map_mid, ctx->label_map_front);
        ctx->have_label_map = false;
    }
    pthread_mutex_unlock(&ctx->labels_swap_mutex);

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return ctx->label_map_front;
}

const uint8_t *
gm_context_get_latest_rgb_label_map(struct gm_context *ctx,
                                    int *width,
                                    int *height)
{
    pthread_mutex_lock(&ctx->labels_swap_mutex);
    if (ctx->have_rgb_label_map) {
        std::swap(ctx->label_map_rgb_mid, ctx->label_map_rgb_front);
        ctx->have_rgb_label_map = false;
    }
    pthread_mutex_unlock(&ctx->labels_swap_mutex);

    *width = ctx->training_camera_intrinsics.width;
    *height = ctx->training_camera_intrinsics.height;

    return ctx->label_map_rgb_front;
}

struct gm_ui_properties *
gm_context_get_ui_properties(struct gm_context *ctx)
{
    return &ctx->properties_state;
}
