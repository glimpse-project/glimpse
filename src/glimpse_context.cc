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

//#define USE_PCL_GBPD 1

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

#ifdef USE_PCL_GBPD
#include <pcl/people/head_based_subcluster.h>
#include <pcl/people/person_classifier.h>
#include <pcl/people/person_cluster.h>
#endif

#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#ifdef __ANDROID__
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#else
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

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "ctx"
#define LOGI(...) gm_info(ctx->log, __VA_ARGS__)
#define LOGE(...) gm_error(ctx->log, __VA_ARGS__)

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

#define CLIPH(X) ((X) > 255 ? 255 : (X))
#define RGB2Y(R, G, B) ((uint8_t)CLIPH(((66 * (uint32_t)(R) + \
                                         129 * (uint32_t)(G) + \
                                         25 * (uint32_t)(B) + 128) >> 8) +  16))

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

enum reproject_op {
    RGBA_INTO_CLOUD,
    RGB_INTO_CLOUD,
    RGB_MIXIN_COPY,
    DEPTH_INTO_BUFFER
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

struct joint_dist
{
    float min;
    float mean;
    float max;
};

struct joint_info
{
    int n_connections;
    int *connections;
    struct joint_dist *dist;
};

struct gm_tracking
{
    struct gm_context *ctx;

    uint64_t depth_capture_timestamp;
    uint64_t video_capture_timestamp;

    // Depth data, in meters
    float *depth;

    // Colour data, RGBA
    uint32_t *video;

    // Depth data mapped to colour for visualisation
    uint8_t *depth_rgb;

    // Label inference data
    uint8_t *label_map;

    // Label inference data in RGB (3x size of label_map) for visualisation
    uint8_t *label_map_rgb;

    // Label probability tables
    float *label_probs;

    // Inferred joint positions
    float *joints;

    // Coloured point cloud
    GlimpsePointXYZRGBA* cloud;
    int cloud_size;

    // Labelled point-cloud for visualisation
    GlimpsePointXYZRGBA* label_cloud;
    int label_cloud_size;

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
    struct gm_extrinsics depth_to_video_extrinsics;
    bool extrinsics_set;

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

    /*
     * -1 means to visualize the most probable labels. Any other value
     *  says to visualize the probability of specific labels...
     */
    int debug_label;

    pthread_mutex_t skel_track_cond_mutex;
    pthread_cond_t skel_track_cond;

    pthread_mutex_t tracking_swap_mutex;
    bool have_tracking;
    struct gm_tracking *tracking_front;
    struct gm_tracking *tracking_mid;
    struct gm_tracking *tracking_back;

    int n_labels;

    JSON_Value *joint_map;
    JIParams *joint_params;
    struct joint_info *joint_stats;
    int n_joints;

    float min_depth;
    float max_depth;

    int n_depth_color_stops;
    float depth_color_stops_range;
    struct color_stop *depth_color_stops;

    int n_heat_color_stops;
    float heat_color_stops_range;
    struct color_stop *heat_color_stops;

    std::vector<struct gm_ui_enumerant> label_enumerants;
    struct gm_ui_properties properties_state;
    std::vector<struct gm_ui_property> properties;

    pthread_mutex_t frame_ready_mutex;
    pthread_cond_t frame_ready_cond;
    struct gm_frame *frame_ready;
    struct gm_frame *frame_front;

    void (*event_callback)(struct gm_context *ctx,
                           struct gm_event *event,
                           void *user_data);

    void *callback_data;

#ifdef USE_PCL_GBPD
    pcl::people::PersonClassifier<pcl::RGB> person_classifier;
#endif
};

static const char *label_names[] = {
    "head, left-side",
    "head, right-side",
    "head, top-left",
    "head, top-right",
    "neck",
    "clavicle, left",
    "clavicle, right",
    "shoulder, left",
    "upper-arm, left",
    "shoulder, right",
    "upper arm, right",
    "elbow, left",
    "forearm, left",
    "elbow, right",
    "forearm, right",
    "wrist, left",
    "hand, left",
    "wrist, right",
    "hand, right",
    "hip, left",
    "thigh, left",
    "hip, right",
    "thigh, right",
    "knee, left",
    "shin, left",
    "knee, right",
    "shin, right",
    "ankle, left",
    "toes, left",
    "ankle, right",
    "toes, right",
    "waist, left",
    "waist, right",
    "background",
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

void
reproject_cloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
                void *buffer,
                const struct gm_intrinsics *intrinsics,
                const struct gm_extrinsics *extrinsics,
                enum reproject_op op,
                GlimpsePointXYZRGBA *cloud_copy = NULL,
                int *cloud_size = NULL,
                bool copy_is_dense = true)
{
    glm::mat3 rotate;
    glm::vec3 translate;

    int width = intrinsics->width;
    int height = intrinsics->height;

    if (cloud_copy && !copy_is_dense) {
        foreach_xy_off(width, height) {
            cloud_copy[off].x = HUGE_DEPTH;
            cloud_copy[off].y = HUGE_DEPTH;
            cloud_copy[off].z = HUGE_DEPTH;
            cloud_copy[off].rgba = 0;
        }
    }
    if (op == DEPTH_INTO_BUFFER) {
        foreach_xy_off(width, height) {
            ((half*)buffer)[off] = HUGE_DEPTH;
        }
    }

    if (extrinsics) {
        const float *r = extrinsics->rotation;
        rotate = glm::mat3(r[0], r[1], r[2],
                           r[3], r[4], r[5],
                           r[6], r[7], r[8]);
        const float *t = extrinsics->translation;
        translate = glm::vec3(t[0], t[1], t[2]);
    }

    /* XXX: we don't assume that cloud->width/height are the same as
     * width/height. The pcl point cloud might not be 2d and if it is
     * we might be projecting a high resolution point cloud into a
     * low resolution depth buffer according to our training camera
     * intrinsics
     */
    int n_points = 0;
    for (uint32_t p = 0; p < cloud->points.size(); p++) {
        pcl::PointXYZRGBA *point = &cloud->points[p];
        glm::vec3 point_t(point->x, point->y, point->z);
        glm::vec2 point_2d;

        if (isnan(point->x) || isinf(point->x) ||
            isnan(point->y) || isinf(point->y) ||
            !isnormal(point->z) || point->z >= HUGE_DEPTH)
            continue;

        if (extrinsics) {
            point_t = (rotate * point_t) + translate;
        }

        int x = (int)
          ((point_t.x * intrinsics->fx / point_t.z) + intrinsics->cx);

        if (x < 0 || x >= width) {
            continue;
        }

        int y = (int)
          ((point_t.y * intrinsics->fy / point_t.z) + intrinsics->cy);

        if (y < 0 || y >= height) {
            continue;
        }

        int off = width * y + x;

        switch(op) {
        case RGBA_INTO_CLOUD:
            point->rgba = ((uint32_t*)buffer)[off];
            break;

        case RGB_INTO_CLOUD: {
            uint8_t r = ((uint8_t*)buffer)[off * 3];
            uint8_t g = ((uint8_t*)buffer)[off * 3 + 1];
            uint8_t b = ((uint8_t*)buffer)[off * 3 + 2];
            uint32_t col = r << 24 | g << 16 | b << 8 | 0xFF;
            point->rgba = col;
            break;
        }

        case DEPTH_INTO_BUFFER:
            ((half*)buffer)[off] = (half)point_t.z;
            break;

        case RGB_MIXIN_COPY:
            break;
        }

        if (cloud_copy) {
            int cloud_off = copy_is_dense ? n_points : off;
            cloud_copy[cloud_off].x = point->x;
            cloud_copy[cloud_off].y = -point->y;
            cloud_copy[cloud_off].z = point->z;

            if (op == RGB_MIXIN_COPY) {
                uint32_t r = (uint32_t)((uint8_t*)buffer)[off * 3];
                uint32_t g = (uint32_t)((uint8_t*)buffer)[off * 3 + 1];
                uint32_t b = (uint32_t)((uint8_t*)buffer)[off * 3 + 2];

                uint32_t r2 = (point->rgba >> 24) & 0xFF;
                uint32_t g2 = (point->rgba >> 16) & 0xFF;
                uint32_t b2 = (point->rgba >> 8) & 0xFF;

                uint32_t col = ((r + r2)/2) << 24 |
                               ((g + g2)/2) << 16 |
                               ((b + b2)/2) << 8 | (point->rgba & 0xFF);

                cloud_copy[cloud_off].rgba = col;

                // To aid visualisation, bump these points a little closer to
                // the camera so they draw over neighbouring points
                cloud_copy[cloud_off].z -= 0.005f;
            } else {
                cloud_copy[cloud_off].rgba = point->rgba;
            }
        }

        ++n_points;
    }

    if (cloud_size) {
        *cloud_size = copy_is_dense ? n_points : (width * height);
    }
}

static bool
sort_indices(pcl::PointIndices a, pcl::PointIndices b) {
    return (a.indices.size() > b.indices.size());
}

static void
tracking_create_rgb_label_map(struct gm_context *ctx,
                              struct gm_tracking *tracking,
                              int debug_label)
{
    int width = ctx->training_camera_intrinsics.width;
    int height = ctx->training_camera_intrinsics.height;
    uint8_t n_labels = ctx->n_labels;
    uint8_t *rgb_label_map = tracking->label_map_rgb;

    gm_assert(ctx->log, debug_label < n_labels,
              "Can't create RGB map of invalid label %u",
              debug_label);

    foreach_xy_off(width, height) {
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

        uint8_t r;
        uint8_t g;
        uint8_t b;

        if (debug_label == -1) {
            r = default_palette[label].red;
            g = default_palette[label].green;
            b = default_palette[label].blue;
        } else {
            struct color col = stops_color_from_val(ctx->heat_color_stops,
                                                    ctx->n_heat_color_stops,
                                                    1,
                                                    pr_table[debug_label]);
            r = col.r;
            g = col.g;
            b = col.b;
        }

        rgb_label_map[pos * 3] = r;
        rgb_label_map[pos * 3 + 1] = g;
        rgb_label_map[pos * 3 + 2] = b;
    }
}

void
gm_context_track_skeleton(struct gm_context *ctx)
{
    struct gm_tracking *tracking = ctx->tracking_back;

    uint64_t start, end, duration;

    // X increases to the right
    // Y increases downwards
    // Z increases outwards

    // Project depth buffer into cloud and filter out points that are too
    // near/far.
    start = get_time();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    cloud->width = 0;
    cloud->height = 1;
    cloud->is_dense = true;

    float inv_fx = 1.0f / ctx->depth_camera_intrinsics.fx;
    float inv_fy = -1.0f / ctx->depth_camera_intrinsics.fy;
    float cx = ctx->depth_camera_intrinsics.cx;
    float cy = ctx->depth_camera_intrinsics.cy;

    cloud->points.resize(ctx->depth_camera_intrinsics.width *
                         ctx->depth_camera_intrinsics.height);
    int n_points = 0;

    // Person detection will happen in a voxel grid of this resolution. A
    // smaller voxel grid may improve results, at the cost of performance.
    float detect_res = 0.06f;
    float half_detect_res = detect_res / 2.f;

    foreach_xy_off(ctx->depth_camera_intrinsics.width,
                   ctx->depth_camera_intrinsics.height) {
        float depth = tracking->depth[off];
        if (isnormal(depth)) {
            float dx = (x - cx) * depth * inv_fx;
            float dy = -(y - cy) * depth * inv_fy;
            cloud->points[n_points].x = dx;
            cloud->points[n_points].y = dy;
            cloud->points[n_points].z = depth;
            cloud->points[n_points].rgba = 0;
            ++n_points;
        }
    }

    cloud->width = n_points;

    end = get_time();
    duration = end - start;
    LOGI("Projection (%d points) took (%.3f%s)\n",
         (int)cloud->points.size(),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Reproject colour information into cloud
    start = get_time();
    reproject_cloud(cloud, (void *)tracking->video,
                    &ctx->video_camera_intrinsics,
                    ctx->extrinsics_set ?
                        &ctx->depth_to_video_extrinsics : NULL,
                    RGBA_INTO_CLOUD,
                    tracking->cloud, &tracking->cloud_size, false);
    end = get_time();
    duration = end - start;
    LOGI("Reprojection of colour information took (%.3f%s)\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Filter out points that are too near or too far
    start = get_time();
    pcl::PassThrough<pcl::PointXYZRGBA> passZ(true);
    passZ.setInputCloud(cloud);
    passZ.setFilterFieldName ("z");
    passZ.setFilterLimits(ctx->min_depth, ctx->max_depth);
    passZ.filter(*cloud);

    end = get_time();
    duration = end - start;
    LOGI("Cloud has %d points after depth filter (%.3f%s)",
         (int)cloud->points.size(),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Simplify point cloud by putting it through a voxel grid
    start = get_time();
    pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(detect_res, detect_res, detect_res);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr dense_cloud(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    vg.filter(*dense_cloud);
    dense_cloud.swap(cloud);

    end = get_time();
    duration = end - start;
    LOGI("Cloud has %d points after voxel grid (%.3f%s)\n",
         (int)cloud->points.size(),
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Detect ground plane.
    // Only consider points below the camera to be the floor.
    // TODO: Rotate point cloud based on device sensors so that the floor is
    //       level. Currently if the camera is pointing downwards, we may end
    //       up filtering out floor points here.
    start = get_time();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_floor(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PassThrough<pcl::PointXYZRGBA> passY;
    passY.setInputCloud(cloud);
    passY.setUserFilterValue(HUGE_DEPTH);
    passY.setKeepOrganized(true);
    passY.setFilterFieldName ("y");
    // XXX: Here we're assuming the camera may be on the ground, but ideally
    //      we'd use a height sensor reading here (minus a threshold).
    passY.setFilterLimits(0.0, FLT_MAX);
    passY.filter(*cloud_floor);

    assert(cloud_floor->points.size() == cloud->points.size());

    int n_floor_points = cloud->points.size() -
      passY.getRemovedIndices()->size();
    LOGI("Cloud possible floor subset has %d points", n_floor_points);

    // Create the segmentation object
    start = get_time();
    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setInputCloud(cloud_floor);

    seg.setMaxIterations(250);
    seg.setDistanceThreshold(std::max(0.03f, detect_res));

    // XXX: We're assuming that the camera here is perpendicular to the floor
    //      and give a generous threshold, but ideally we'd use device sensors
    //      to detect orientation and use a slightly less broad angle here.
    seg.setAxis(Eigen::Vector3f(0.f, -1.f, 0.f));
    seg.setEpsAngle(M_PI/180.0 * 15.0);

    pcl::ModelCoefficients::Ptr ground_coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.segment(*inliers, *ground_coeffs);

    int n_inliers = (int)inliers->indices.size();

    end = get_time();
    duration = end - start;
    LOGI("Looking for ground plane took (%.3f%s)\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    if (n_inliers >= (int)(n_floor_points * 0.25f)) {
        // Remove ground plane
        start = get_time();
        pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud);

        end = get_time();
        duration = end - start;
        LOGI("Ground plane has %d points (%.3f%s)\n",
             n_inliers,
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));
    }

    // Use Euclidean clustering to split the cloud into possible human clusters.
    start = get_time();

    // The search method used depends a fair bit on the density of the cloud
    // being searched and the number of searches being performed. A KdTree is
    // usually faster for this particular use-case than an Octree, but this can
    // vary depending on situation and compile options.
#if 1
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr
      tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
#else
    pcl::search::Octree<pcl::PointXYZRGBA>::Ptr
      tree(new pcl::search::Octree<pcl::PointXYZRGBA>(0.1));
#endif

    tree->setSortedResults(false);
    tree->setInputCloud(cloud);

    // Note that these values don't really correspond to human size like you
    // might think. This is what PCL does, and it's as good a measure as any,
    // but take it with a huge pinch of salt.
    float min_height = 1.3;
    float max_height = 2.15;
    float min_width = 0.3;
    float max_width = 1.5;

    int min_points = (int)(min_height * min_width / detect_res / detect_res);
    int max_points = (int)(max_height * max_width / detect_res / detect_res);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    ec.setClusterTolerance(std::max(0.03f, detect_res * 2));
    ec.setMinClusterSize(min_points);
    ec.setMaxClusterSize(max_points);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    end = get_time();
    duration = end - start;
    LOGI("Euclidean clustering took (%.3f%s)\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    start = get_time();
    bool person_found = false;
    pcl::PointIndices person_indices;

    // Create a search tree so we can reconstruct the dense person cloud
    // We use an octree here as the time it takes to build a KdTree vastly
    // outweighs the speed benefit searching it.
    pcl::search::Octree<pcl::PointXYZRGBA>::Ptr
      dense_tree(new pcl::search::Octree<pcl::PointXYZRGBA>(detect_res));
    dense_tree->setSortedResults(false);
    dense_tree->setInputCloud(dense_cloud);

#ifndef USE_PCL_GBPD
    // Assume the largest cluster that contains its centroid may be a person

    std::sort(cluster_indices.begin(), cluster_indices.end(), sort_indices);

    for (std::vector<pcl::PointIndices>::const_iterator point_it =
            cluster_indices.begin();
         point_it != cluster_indices.end(); ++point_it) {
        const pcl::PointIndices cluster = *point_it;

        // Work out the centroid of the cloud and see if there's a point
        // near there. A human, unless they're falling, ought to contain
        // their center of gravity. If they're jumping or falling, we can
        // probably interpolate joint positions (TODO: Interpolation...)
        Eigen::VectorXf centroid;
        pcl::computeNDCentroid(*cloud, *point_it, centroid);

        // Make sure the centroid is in the mass
        std::vector<int>centroid_indices;
        std::vector<float>centroid_distances;
        pcl::PointXYZRGBA centroid_pt;
        centroid_pt.x = centroid[0];
        centroid_pt.y = centroid[1];
        centroid_pt.z = centroid[2];
        dense_tree->radiusSearch(centroid_pt, detect_res * 2,
                                centroid_indices, centroid_distances);

        if (centroid_indices.size() <= 0) {
            continue;
        }

        person_indices = *point_it;
        person_found = true;
        break;
    }
#else
    // Use pcl's HeadBasedSubclustering and PersonClassifier
    // We do this because pcl doesn't call setDimensionLimits on the
    // HeadBasedSubclustering object, and also it makes it easier to integrate
    // and toggle it off/on for comparison.
    Eigen::VectorXf eigen_ground_coeffs;
    eigen_ground_coeffs.resize(4);
    eigen_ground_coeffs[0] = ground_coeffs->values[0];
    eigen_ground_coeffs[1] = ground_coeffs->values[1];
    eigen_ground_coeffs[2] = ground_coeffs->values[2];
    eigen_ground_coeffs[3] = ground_coeffs->values[3];

    pcl::people::HeadBasedSubclustering<pcl::PointXYZRGBA> subclustering;
    subclustering.setDimensionLimits(min_points, max_points);
    subclustering.setInputCloud(cloud);
    subclustering.setGround(eigen_ground_coeffs);
    subclustering.setInitialClusters(cluster_indices);
    subclustering.setHeightLimits(min_height, max_height);

    std::vector<pcl::people::PersonCluster<pcl::PointXYZRGBA>> clusters;
    subclustering.subcluster(clusters);

    // Use the person classifier to add confidence to the person clusters
    pcl::PointCloud<pcl::RGB>::Ptr rgb_image(new pcl::PointCloud<pcl::RGB>);
    rgb_image->width = ctx->depth_camera_intrinsics.width;
    rgb_image->height = ctx->depth_camera_intrinsics.height;
    rgb_image->points.resize(rgb_image->width * rgb_image->height);
    foreach_xy_off(rgb_image->width, rgb_image->height) {
        int target_off = (rgb_image->height - y - 1) * rgb_image->width + x;
        rgb_image->points[target_off].rgb = tracking->video[off] >> 8;
    }

    Eigen::Matrix3f eigen_intrinsics;
    eigen_intrinsics <<
        (float)ctx->video_camera_intrinsics.fx, 0.f,
        (float)ctx->video_camera_intrinsics.fy, 0.f,
        (float)ctx->video_camera_intrinsics.cx,
        (float)ctx->video_camera_intrinsics.cy,
        0.f, 0.f, 0.1f;

    for (std::vector<pcl::people::PersonCluster<pcl::PointXYZRGBA>>::iterator it = clusters.begin();
        it != clusters.end(); ++it) {
        // Evaluate confidence for the current PersonCluster:
        Eigen::Vector3f centroid = eigen_intrinsics * (it->getTCenter());
        centroid /= centroid(2);

        Eigen::Vector3f top = eigen_intrinsics * (it->getTTop());
        top /= top(2);

        Eigen::Vector3f bottom = eigen_intrinsics * (it->getTBottom());
        bottom /= bottom(2);

        it->setPersonConfidence(ctx->person_classifier.evaluate(
            rgb_image, bottom, top, centroid, false));
    }

    float max_confidence = -FLT_MAX;
    for (std::vector<pcl::people::PersonCluster<pcl::PointXYZRGBA>>::iterator
        it = clusters.begin(); it != clusters.end(); ++it) {
        float confidence = it->getPersonConfidence();
        LOGI("Person cluster with %f", confidence);

        if (confidence <= max_confidence) { continue; }

        person_indices = it->getIndices();
        person_found = true;
        max_confidence = confidence;
    }
#endif

    end = get_time();
    duration = end - start;
    LOGI("People detection took %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    if (!person_found) {
        LOGE("Skipping detection: Could not find a person cluster\n");
        return;
    }

    // Do a radius search on each point from the sparse tree to rebuild the
    // dense cloud cluster.
    start = get_time();

    std::list<int> indices;
    for (std::vector<int>::const_iterator it = person_indices.indices.begin();
         it != person_indices.indices.end (); ++it) {
        std::vector<int> pt_indices;

        Eigen::Vector3f pt_min, pt_max;
        pcl::PointXYZRGBA point = cloud->points[*it];
        pt_min[0] = point.x - half_detect_res;
        pt_min[1] = point.y - half_detect_res;
        pt_min[2] = point.z - half_detect_res;
        pt_max[0] = point.x + half_detect_res;
        pt_max[1] = point.y + half_detect_res;
        pt_max[2] = point.z + half_detect_res;

        dense_tree->tree_->boxSearch(pt_min, pt_max, pt_indices);
        std::copy(pt_indices.begin(), pt_indices.end(),
                  std::back_inserter(indices));
    }
    indices.sort();
    indices.unique();

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr person_cluster(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    for (std::list<int>::const_iterator it = indices.begin();
         it != indices.end (); ++it) {
        person_cluster->points.push_back(dense_cloud->points[*it]);
    }
    person_cluster->width = indices.size();
    person_cluster->height = 1;
    person_cluster->is_dense = true;

    cloud.swap(person_cluster);

    end = get_time();
    duration = end - start;
    LOGI("Person reconstruction (%d points) took (%.3f%s)\n",
         (int)person_cluster->width,
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
    int width = ctx->training_camera_intrinsics.width;
    int height = ctx->training_camera_intrinsics.height;
    half *depth_img = (half *)xmalloc(width * height * sizeof(half));

    reproject_cloud(cloud, (void *)depth_img, &ctx->training_camera_intrinsics,
                    NULL, DEPTH_INTO_BUFFER);

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
        LOGE("People detector: exiting: No decision trees loaded\n");
        return;
    }
    LOGI("People Detector: starting label inference: n_trees=%d, w=%d, h=%d, data=%p",
         ctx->n_decision_trees, width, height, depth_img);
    start = get_time();
    infer_labels(ctx->decision_trees,
                 ctx->n_decision_trees,
                 depth_img,
                 width,
                 height,
                 tracking->label_probs);
    end = get_time();
    duration = end - start;
    LOGI("People Detector: ran label probability inference in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    start = get_time();
    float *weights = calc_pixel_weights(depth_img,
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
        InferredJoints *result =
            infer_joints_fast(depth_img,
                              tracking->label_probs,
                              weights,
                              width, height,
                              ctx->n_labels,
                              ctx->joint_map,
                              vfov,
                              ctx->joint_params->joint_params);

        assert(result->n_joints = ctx->n_joints);
        LList *joint_ptrs[ctx->n_joints];
        for (int j = 0; j < ctx->n_joints; j++) {
            if (result->joints[j]) {
                Joint *joint = (Joint *)result->joints[j]->data;
                tracking->joints[j*3] = joint->x;
                tracking->joints[j*3+1] = joint->y;
                tracking->joints[j*3+2] = joint->z;
                joint_ptrs[j] = result->joints[j];
            }
        }
        // Try to see if there are better joint predictions based on statistics
        // of distances between joints.
        if (ctx->joint_stats) {
            // XXX: Maybe put a limit on the number of passes this does(?)
            //      It's pretty inexpensive though.
            bool changed;
            do {
                changed = false;
                for (int j = 0; j < ctx->n_joints; j++) {
                    // If we have < 2 joint candidates, skip over this joint.
                    if (!result->joints[j] || !result->joints[j]->next) {
                        continue;
                    }

                    LList *best_joint = NULL;
                    float best_dist = FLT_MAX;
                    int best_error = INT_MAX;

                    for (LList *l = result->joints[j]; l; l = l->next) {
                        Joint *joint = (Joint *)l->data;
                        float dist_acc = 0;
                        int error = 0;
                        for (int k = 0; k < ctx->joint_stats[j].n_connections;
                             k++) {
                            struct joint_dist *joint_dist =
                                &ctx->joint_stats[j].dist[k];
                            int c = ctx->joint_stats[j].connections[k];
                            float dist = sqrtf(
                                powf(tracking->joints[c*3] - joint->x, 2.f) +
                                powf(tracking->joints[c*3+1] - joint->y, 2.f) +
                                powf(tracking->joints[c*3+2] - joint->z, 2.f));
                            dist = fabsf(dist - joint_dist->mean);
                            dist_acc += dist;
                            if (dist < joint_dist->min ||
                                dist > joint_dist->max) {
                                ++error;
                            }
                        }
                        if (error < best_error && dist_acc <= best_dist) {
                            best_dist = dist_acc;
                            best_error = error;
                            best_joint = l;
                        }
                    }

                    if (best_joint && joint_ptrs[j] != best_joint) {
                        // Null out the original data so we don't end up with
                        // situations where we oscillate between two choices
                        Joint *joint = (Joint *)joint_ptrs[j]->data;
                        joint->x = 0.f;
                        joint->y = 0.f;
                        joint->z = HUGE_DEPTH * 2;

                        joint_ptrs[j] = best_joint;
                        joint = (Joint *)best_joint->data;
                        tracking->joints[j*3] = joint->x;
                        tracking->joints[j*3+1] = joint->y;
                        tracking->joints[j*3+2] = joint->z;
                        changed = true;
                    }
                }
            } while (changed);
        }
        free_joints(result);

        end = get_time();
        duration = end - start;
        LOGI("People Detector: inferred joints in %.3f%s\n",
             get_duration_ns_print_scale(duration),
             get_duration_ns_print_scale_suffix(duration));
    }
#endif
    free(weights);
    weights = NULL;
    free(depth_img);
    depth_img = NULL;

    start = get_time();
    tracking_create_rgb_label_map(ctx, tracking, ctx->debug_label);
    end = get_time();
    duration = end - start;
    LOGI("Created RGB label map in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));

    // Colourise cloud based on labels for debug
    start = get_time();

    // This tints the coloured depth cloud with the inferred label colours,
    // but turns out not to be that useful a visualisation due to the
    // sparseness of the label cloud vs. the original depth cloud.
    /*reproject_cloud(cloud, (void *)rgb_label_map,
                    &ctx->training_camera_intrinsics,
                    ctx->extrinsics_set ?
                        &ctx->depth_to_video_extrinsics : NULL,
                    RGB_MIXIN_COPY,
                    tracking->cloud, NULL, false);*/

    reproject_cloud(cloud, (void*)tracking->label_map_rgb,
                    &ctx->training_camera_intrinsics,
                    NULL,
                    RGB_INTO_CLOUD,
                    tracking->label_cloud,
                    &tracking->label_cloud_size,
                    true);

    end = get_time();
    duration = end - start;
    LOGI("Reprojected RGB label map in %.3f%s\n",
         get_duration_ns_print_scale(duration),
         get_duration_ns_print_scale_suffix(duration));
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
update_tracking_video(struct gm_context *ctx,
                      struct gm_tracking *tracking,
                      enum gm_format format,
                      uint8_t *video,
                      uint64_t timestamp)
{
    int width = ctx->video_camera_intrinsics.width;
    int height = ctx->video_camera_intrinsics.height;

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
    case GM_FORMAT_RGB:
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 3];
            uint8_t g = video[off * 3 + 1];
            uint8_t b = video[off * 3 + 2];
            tracking->video[off] = (r << 24) | (g << 16) | (b << 8) | 0xFF;
            tracking->face_detect_buf[off] = RGB2Y(r, g, b);
        }
        break;

    case GM_FORMAT_RGBX:
        foreach_xy_off(width, height) {
            uint32_t rgba = ((uint32_t*)video)[off] | 0xFF;
            tracking->video[off] = rgba;

            uint8_t r = video[off * 3];
            uint8_t g = video[off * 3 + 1];
            uint8_t b = video[off * 3 + 2];
            tracking->face_detect_buf[off] = RGB2Y(r, g, b);
        }
        break;

    case GM_FORMAT_RGBA:
        memcpy(tracking->video, video, width * height * 4);
        foreach_xy_off(width, height) {
            uint8_t r = video[off * 3];
            uint8_t g = video[off * 3 + 1];
            uint8_t b = video[off * 3 + 2];
            tracking->face_detect_buf[off] = RGB2Y(r, g, b);
        }
        break;

    case GM_FORMAT_LUMINANCE_U8:
        foreach_xy_off(width, height) {
            uint8_t lum = video[off];
            tracking->video[off] = (lum << 24) | (lum << 16) | (lum << 8) | 0xFF;
        }
        memcpy(tracking->face_detect_buf, video, width * height);
        break;

    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_Z_U16_MM:
    case GM_FORMAT_Z_F32_M:
    case GM_FORMAT_Z_F16_M:
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
    int width = ctx->depth_camera_intrinsics.width;
    int height = ctx->depth_camera_intrinsics.height;

#define COPY_AND_MAP_DEPTH_TO_RGB(DEPTH_COPY, DEPTH_RGB_BUF, OFF, DEPTH) \
    do { \
        struct color rgb = stops_color_from_val(ctx->depth_color_stops, \
                                                ctx->n_depth_color_stops, \
                                                ctx->depth_color_stops_range, \
                                                DEPTH); \
        uint8_t *depth_rgb = DEPTH_RGB_BUF + OFF * 3; \
        depth_rgb[0] = rgb.r; \
        depth_rgb[1] = rgb.g; \
        depth_rgb[2] = rgb.b; \
        depth_copy[OFF] = DEPTH; \
    } while (0)

    float *depth_copy = tracking->depth;
    uint8_t *depth_rgb_back = tracking->depth_rgb;

    switch (format) {
    case GM_FORMAT_Z_U16_MM:
        foreach_xy_off(width, height) {
            float depth_m = ((uint16_t *)depth)[off] / 1000.f;
            COPY_AND_MAP_DEPTH_TO_RGB(depth_copy, depth_rgb_back, off, depth_m);
        }
        break;
    case GM_FORMAT_Z_F32_M:
        foreach_xy_off(width, height) {
            float depth_m = ((float *)depth)[off];
            COPY_AND_MAP_DEPTH_TO_RGB(depth_copy, depth_rgb_back, off, depth_m);
        }
        break;
    case GM_FORMAT_Z_F16_M:
        foreach_xy_off(width, height) {
            float depth_m = ((half *)depth)[off];
            COPY_AND_MAP_DEPTH_TO_RGB(depth_copy, depth_rgb_back, off, depth_m);
        }
        break;
    case GM_FORMAT_UNKNOWN:
    case GM_FORMAT_LUMINANCE_U8:
    case GM_FORMAT_RGB:
    case GM_FORMAT_RGBX:
    case GM_FORMAT_RGBA:
        gm_assert(ctx->log, 0, "Unexpected format for depth buffer");
        break;
    }

#undef COPY_AND_MAP_DEPTH_TO_RGB

    tracking->depth_capture_timestamp = timestamp;
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
        if (ctx->frame_front)
            gm_frame_unref(ctx->frame_front);
        ctx->frame_front = ctx->frame_ready;
        ctx->frame_ready = NULL;
        pthread_mutex_unlock(&ctx->frame_ready_mutex);

        struct gm_frame *frame = ctx->frame_front;
        update_tracking_depth_from_buffer(ctx,
                                          ctx->tracking_back,
                                          frame->depth_format,
                                          frame->depth->data,
                                          frame->timestamp);

        update_tracking_video(ctx,
                              ctx->tracking_back,
                              frame->video_format,
                              (uint8_t *)frame->video->data,
                              frame->timestamp);

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
    free(tracking->label_map_rgb);
    free(tracking->label_probs);
    free(tracking->joints);
    free(tracking->label_cloud);

    free(tracking->depth);
    free(tracking->depth_rgb);
    free(tracking->cloud);

    free(tracking->video);
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

    tracking->label_map_rgb = (uint8_t *)xcalloc(labels_width * labels_height, 3);
    tracking->label_probs = (float *)xcalloc(labels_width *
                                             labels_height *
                                             sizeof(float) * ctx->n_labels, 1);
    tracking->joints = (float *)xcalloc(ctx->n_joints, 3 * sizeof(float));
    tracking->label_cloud = (GlimpsePointXYZRGBA *)
        xcalloc(labels_width * labels_height, sizeof(GlimpsePointXYZRGBA));

    int depth_width = ctx->depth_camera_intrinsics.width;
    int depth_height = ctx->depth_camera_intrinsics.height;

    assert(depth_width);
    assert(depth_height);

    tracking->depth = (float *)
      xcalloc(depth_width * depth_height, sizeof(float));
    tracking->depth_rgb = (uint8_t *)xcalloc(depth_width * depth_height, 3);
    tracking->cloud = (GlimpsePointXYZRGBA *)
        xcalloc(depth_width * depth_height, sizeof(GlimpsePointXYZRGBA));

    int video_width = ctx->video_camera_intrinsics.width;
    int video_height = ctx->video_camera_intrinsics.height;

    tracking->video = (uint32_t *)
      xcalloc(video_width * video_height, sizeof(uint32_t));

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

    return tracking;
}


void
gm_context_destroy(struct gm_context *ctx)
{
    free_tracking(ctx->tracking_front);
    free_tracking(ctx->tracking_mid);
    free_tracking(ctx->tracking_back);

    free(ctx->depth_color_stops);

    for (int i = 0; i < ctx->n_decision_trees; i++)
        free_tree(ctx->decision_trees[i]);
    xfree(ctx->decision_trees);

    free_jip(ctx->joint_params);

    json_value_free(ctx->joint_map);

    if (ctx->joint_stats) {
        for (int i = 0; i < ctx->n_joints; i++) {
            xfree(ctx->joint_stats[i].connections);
            xfree(ctx->joint_stats[i].dist);
        }
        xfree(ctx->joint_stats);
    }

    delete ctx;
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

    // Load joint statistics for improving the quality of predicted joint
    // positions.
    ctx->joint_stats = NULL;

    struct gm_asset *joint_stats_asset =
        gm_asset_open("joint-dist.json", GM_ASSET_MODE_BUFFER, &open_err);
    if (joint_stats_asset) {
        ctx->joint_stats = (struct joint_info *)
            xcalloc(ctx->n_joints, sizeof(struct joint_info));
        for (int i = 0; i < ctx->n_joints; i++) {
            ctx->joint_stats[i].connections = (int *)
                xmalloc(ctx->n_joints * sizeof(int));
            ctx->joint_stats[i].dist = (struct joint_dist *)
                xmalloc(ctx->n_joints * sizeof(struct joint_dist));
        }

        // Discover joint connections
        for (int i = 0; i < ctx->n_joints; i++) {
            JSON_Object *joint =
                json_array_get_object(json_array(ctx->joint_map), i);
            JSON_Array *connections =
                json_object_get_array(joint, "connections");
            for (int c = 0; c < json_array_get_count(connections); c++) {
                const char *name = json_array_get_string(connections, c);
                for (int j = 0; j < ctx->n_joints; j++) {
                    JSON_Object *connection =
                        json_array_get_object(json_array(ctx->joint_map), j);
                    if (strcmp(json_object_get_string(connection, "joint"),
                               name) != 0) { continue; }

                    // Add the connection to this joint and add the reverse
                    // connection.
                    int idx = ctx->joint_stats[i].n_connections;
                    ctx->joint_stats[i].connections[idx] = j;
                    ++ctx->joint_stats[i].n_connections;

                    idx = ctx->joint_stats[j].n_connections;
                    ctx->joint_stats[j].connections[idx] = i;
                    ++ctx->joint_stats[j].n_connections;

                    break;
                }
            }
        }

        const void *buf = gm_asset_get_buffer(joint_stats_asset);
        JSON_Value *json = json_parse_string((char *)buf);

        assert(json_array_get_count(json_array(json)) == ctx->n_joints);
        for (int i = 0; i < ctx->n_joints; i++) {
            JSON_Array *stats = json_array_get_array(json_array(json), i);
            assert(json_array_get_count(stats) == ctx->n_joints);

            for (int j = 0; j < ctx->joint_stats[i].n_connections; i++) {
                int c = ctx->joint_stats[i].connections[j];
                JSON_Object *stat = json_array_get_object(stats, c);
                ctx->joint_stats[i].dist[j].min = (float)
                    json_object_get_number(stat, "min");
                ctx->joint_stats[i].dist[j].mean = (float)
                    json_object_get_number(stat, "mean");
                ctx->joint_stats[i].dist[j].max = (float)
                    json_object_get_number(stat, "max");
            }
        }

        gm_asset_close(joint_stats_asset);
        json_value_free(json);
    } else {
        xasprintf(err, "Failed to open joint-dist.json: %s", open_err);
        free(open_err);

        // We can continue without the joint stats asset, just results may be
        // poorer quality.
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

#ifdef USE_PCL_GBPD
    struct gm_asset *svm_asset = gm_asset_open("person-hog-svm.json",
                                               GM_ASSET_MODE_BUFFER,
                                               &open_err);
    if (svm_asset) {
        // XXX: We're assuming svm.json is well-formed (we provide it, so it is)
        const void *buf = gm_asset_get_buffer(svm_asset);
        JSON_Value *svm_json = json_parse_string((char *)buf);
        JSON_Object *svm_root = json_value_get_object(svm_json);

        gm_asset_close(svm_asset);
        buf = NULL;
        svm_asset = NULL;

        std::vector<float> weights;
        JSON_Array *weights_array = json_object_get_array(svm_root, "weights");
        for (size_t i = 0; i < json_array_get_count(weights_array); i++) {
            weights.push_back((float)json_array_get_number(weights_array, i));
        }

        ctx->person_classifier.setSVM(
            (int)json_object_get_number(svm_root, "window_height"),
            (int)json_object_get_number(svm_root, "window_width"),
            weights,
            (float)json_object_get_number(svm_root, "offset"));

        json_value_free(svm_json);
    } else {
        xasprintf(err, "Failed to open svm.json: %s", open_err);
        free(open_err);
        gm_context_destroy(ctx);
        return NULL;
    }
#endif

    struct gm_ui_property prop;

    ctx->min_depth = 0.5;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "min_depth";
    prop.desc = "throw away points nearer than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->min_depth;
    prop.float_state.min = 0.0;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->max_depth = 3.5;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "max_depth";
    prop.desc = "throw away points further than this";
    prop.type = GM_PROPERTY_FLOAT;
    prop.float_state.ptr = &ctx->max_depth;
    prop.float_state.min = 0.5;
    prop.float_state.max = 10;
    ctx->properties.push_back(prop);

    ctx->debug_label = -1;
    prop = gm_ui_property();
    prop.object = ctx;
    prop.name = "debug_label";
    prop.desc = "visualize specific label probabilities";
    prop.type = GM_PROPERTY_INT;
    prop.enum_state.ptr = &ctx->debug_label;

    struct gm_ui_enumerant enumerant;
    enumerant = gm_ui_enumerant();
    enumerant.name = "most likely";
    enumerant.desc = "Visualize Most Probable Labels";
    prop.type = GM_PROPERTY_ENUM;
    enumerant.val = -1;
    ctx->label_enumerants.push_back(enumerant);

    gm_assert(ctx->log,
              ctx->n_labels == ARRAY_LEN(label_names),
              "Mismatched label name count");

    for (int i = 0; i < ctx->n_labels; i++) {
        enumerant = gm_ui_enumerant();
        enumerant.name = label_names[i];
        enumerant.desc = label_names[i];
        enumerant.val = i;
        ctx->label_enumerants.push_back(enumerant);
    }
    prop.enum_state.n_enumerants = ctx->label_enumerants.size();
    prop.enum_state.enumerants = ctx->label_enumerants.data();
    ctx->properties.push_back(prop);

    ctx->properties_state.n_properties = ctx->properties.size();
    pthread_mutex_init(&ctx->properties_state.lock, NULL);
    ctx->properties_state.properties = &ctx->properties[0];

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
gm_context_set_depth_to_video_camera_extrinsics(struct gm_context *ctx,
                                                struct gm_extrinsics *extrinsics)
{
    if (extrinsics) {
        ctx->depth_to_video_extrinsics = *extrinsics;
        ctx->extrinsics_set = true;
    } else {
        ctx->extrinsics_set = false;
    }
}

const uint8_t *
gm_tracking_get_rgb_depth(struct gm_tracking *tracking)
{
    //struct gm_context *ctx = tracking->ctx;

    return tracking->depth_rgb;
}

const GlimpsePointXYZRGBA *
gm_tracking_get_rgb_cloud(struct gm_tracking *tracking,
                          int *n_points)
{
    if (n_points)
      *n_points = tracking->cloud_size;
    return tracking->cloud;
}

const GlimpsePointXYZRGBA *
gm_tracking_get_rgb_label_cloud(struct gm_tracking *tracking,
                                int *n_points)
{
    if (n_points)
      *n_points = tracking->label_cloud_size;
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

    pthread_mutex_lock(&ctx->frame_ready_mutex);
    if (ctx->frame_ready)
        gm_frame_unref(ctx->frame_ready);
    ctx->frame_ready = gm_frame_ref(frame);
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

