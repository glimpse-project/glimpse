#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <locale.h>
#include <assert.h>

#include <math.h>

#include <pthread.h>

#include <epoxy/gl.h>
#include <epoxy/egl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <libfreenect.h>

#include "glimpse_context.h"

#include <imgui.h>
#include <imgui_impl_glfw_gles3.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define TOOLBAR_LEFT_WIDTH 300

typedef struct _Data
{
    GLFWwindow *window;
    int win_width;
    int win_height;

    int attr_pos;
    int attr_tex_coord;
    int attr_color;

} Data;


static struct gm_context *gm_context;

/*
 * We generally use tripple buffering to decouple the capture and preparation
 * of new data buffers from the buffers used to render.
 *
 * Preparation happens within the back buffer and when it's ready it
 * gets swapped with the mid buffer and a flag is set to indicate that
 * new data is available.
 *
 * The render thread can always render the front buffer. The renderer
 * regularly checks if there's a new valid mid buffer and if so it swaps
 * it with the front buffer.
 */
static uint8_t *depth_rgb_back, *depth_rgb_mid, *depth_rgb_front;
static bool depth_rgb_mid_valid;

static uint8_t *rgb_back, *rgb_mid, *rgb_front;
static uint8_t *lum_back, *lum_mid, *lum_front;
static bool rgb_mid_valid; // also covers lum_mid

static pthread_mutex_t swap_buffers_mutex = PTHREAD_MUTEX_INITIALIZER;

static GLuint gl_tex_program;
static GLuint uniform_tex_sampler;
static GLuint gl_labels_tex;
static GLuint gl_depth_rgb_tex;
static GLuint gl_rgb_tex;
static GLuint gl_lum_tex;

static freenect_context *kinect_ctx;
static freenect_device *kinect_dev;

static int kinect_ir_brightness;
static float kinect_tilt;
static float kinect_accel_x;
static float kinect_accel_y;
static float kinect_accel_z;
static float kinect_mks_accel_x;
static float kinect_mks_accel_y;
static float kinect_mks_accel_z;

static pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;

static void
redraw(Data *data)
{
    gm_context_render_thread_hook(gm_context);


    float left_col = TOOLBAR_LEFT_WIDTH;
    ImVec2 main_menu_size;
    ImVec2 win_size;

    ImGui_ImplGlfwGLES3_NewFrame();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

    if (ImGui::BeginMainMenuBar()) {
        main_menu_size = ImGui::GetWindowSize();

        if (ImGui::BeginMenu("File")) {

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::SetNextWindowPos(ImVec2(0, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(left_col, data->win_height));
    ImGui::Begin("Controls", NULL,
                 ImGuiWindowFlags_NoTitleBar|
                 ImGuiWindowFlags_NoResize);
    if (ImGui::SliderInt("IR Intensity", &kinect_ir_brightness, 1, 50))
        freenect_set_ir_brightness(kinect_dev, kinect_ir_brightness);
    if (ImGui::SliderFloat("Tilt", &kinect_tilt, -30, 30))
        freenect_set_tilt_degs(kinect_dev, kinect_tilt);

    ImGui::Separator();
    ImGui::LabelText("Accel", "%.3f,%.3f,%.3f",
                     kinect_accel_x,
                     kinect_accel_y,
                     kinect_accel_z);
    ImGui::LabelText("MKS Accel", "%.3f,%.3f,%.3f",
                     kinect_mks_accel_x,
                     kinect_mks_accel_y,
                     kinect_mks_accel_z);

    struct gm_ui_properties *props = gm_context_get_ui_properties(gm_context);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        if (prop->type == GM_PROPERTY_INT)
            ImGui::SliderInt(prop->name, prop->int_ptr, prop->min, prop->max);
        if (prop->type == GM_PROPERTY_FLOAT)
            ImGui::SliderFloat(prop->name, prop->float_ptr, prop->min, prop->max);
    }

    ImGui::End();

    ImVec2 main_area_size = ImVec2(data->win_width - left_col, data->win_height - main_menu_size.y);

    ImGui::SetNextWindowPos(ImVec2(left_col, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Depth Buffer", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_depth_rgb_tex, win_size);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(left_col, main_menu_size.y + main_area_size.y/2));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Luminance", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_lum_tex, win_size);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(left_col + main_area_size.x/2, main_menu_size.y));
    ImGui::SetNextWindowSize(ImVec2(main_area_size.x/2, main_area_size.y/2));
    ImGui::Begin("Labels", NULL,
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoResize);
    win_size = ImGui::GetWindowSize();
    ImGui::Image((void *)(intptr_t)gl_labels_tex, win_size);
    ImGui::End();

    ImGui::PopStyleVar();

    /* Handy to look at the various features of imgui... */
#if 0
    ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
    ImGui::ShowTestWindow(&show_test_window);
#endif

    /*
     * Upload textures
     */


    glViewport(0, 0, data->win_width, data->win_height);
    glClear(GL_COLOR_BUFFER_BIT);

    pthread_mutex_lock(&swap_buffers_mutex);

    if (depth_rgb_mid_valid) {
        //std::swap(depth_front, depth_mid);
        std::swap(depth_rgb_front, depth_rgb_mid);
        depth_rgb_mid_valid = false;
    }
    if (rgb_mid_valid) {
        std::swap(rgb_front, rgb_mid);
        std::swap(lum_front, lum_mid);
        rgb_mid_valid = false;
    }

    pthread_mutex_unlock(&swap_buffers_mutex);

    /*
     * Draw the depth buffer via RGB
     */
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, depth_rgb_front);

    /*
     * Draw luminance from RGB camera
     */
    glBindTexture(GL_TEXTURE_2D, gl_lum_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                 640, 480, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, lum_front);

    /*
     * Draw inferred label map
     */
    glBindTexture(GL_TEXTURE_2D, gl_labels_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    int label_map_width = 0;
    int label_map_height = 0;
    const uint8_t *labels_rgb = gm_context_get_latest_rgb_label_map(gm_context,
                                                                    &label_map_width,
                                                                    &label_map_height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 label_map_width, label_map_height, 0, GL_RGB, GL_UNSIGNED_BYTE, labels_rgb);

    ImGui::Render();
    glfwSwapBuffers(data->window);
}

static void
init_opengl(Data *data)
{
    static const char *vertShaderText =
        "#version 300 es\n"
        "precision mediump float;\n"
        "precision mediump int;\n"
        "in vec4 pos;\n"
        "in vec4 color;\n"
        "in vec4 tex_coord;\n"
        "out vec4 v_tex_coord;\n"
        "out vec4 v_color;\n"
        "\n"
        "void main () {\n"
        "  gl_Position = pos;\n"
        "  v_tex_coord = tex_coord;\n"
        "  v_color = color;\n"
        "}\n";

    static const char *fragShaderText =
        "#version 300 es\n"
        "precision mediump float;\n"
        "precision mediump int;\n"
        "in vec4 v_tex_coord;\n"
        "in vec4 v_color;\n"
        "out vec4 frag_color;\n"
        "uniform sampler2D texture;\n"
        "\n"
        "void main () {\n"
        "  frag_color = texture2D(texture, v_tex_coord.st) * v_color;\n"
        "}\n";

    GLuint fragShader, vertShader;
    GLint stat;

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClearStencil(0);

    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, (const char **) &fragShaderText, NULL);
    glCompileShader(fragShader);
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &stat);
    if (!stat) {
        char log[1000];
        GLsizei len;
        glGetShaderInfoLog (fragShader, 1000, &len, log);
        printf("Error: fragment shader did not compile: %s\n", log);
        exit(1);
    }

    vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, (const char **) &vertShaderText, NULL);
    glCompileShader(vertShader);
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &stat);
    if (!stat) {
        char log[1000];
        GLsizei len;
        glGetShaderInfoLog(vertShader, 1000, &len, log);
        printf("Error: vertex shader did not compile: %s\n", log);
        exit(1);
    }

    gl_tex_program = glCreateProgram();
    glAttachShader(gl_tex_program, fragShader);
    glAttachShader(gl_tex_program, vertShader);
    glLinkProgram(gl_tex_program);

    glGetProgramiv(gl_tex_program, GL_LINK_STATUS, &stat);
    if (!stat) {
        char log[1000];
        GLsizei len;
        glGetProgramInfoLog(gl_tex_program, 1000, &len, log);
        printf ("Error: linking:\n%s\n", log);
        exit (1);
    }

    glUseProgram(gl_tex_program);

    uniform_tex_sampler = glGetUniformLocation(gl_tex_program, "texture");
    glUniform1i(uniform_tex_sampler, 0);

    glBindAttribLocation(gl_tex_program, 0, "pos");
    data->attr_pos = 0;

    //data->attr_pos = glGetAttribLocation (gl_tex_program, "pos");
    data->attr_tex_coord = glGetAttribLocation(gl_tex_program, "tex_coord");
    data->attr_color = glGetAttribLocation(gl_tex_program, "color");

    glUseProgram(0);

    glGenTextures(1, &gl_depth_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_lum_tex);
    glBindTexture(GL_TEXTURE_2D, gl_lum_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_labels_tex);
    glBindTexture(GL_TEXTURE_2D, gl_labels_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void
event_loop(Data *data)
{
    while (!glfwWindowShouldClose(data->window)) {
        glfwPollEvents();
        redraw(data);
    }
}

static void
kinect_depth_frame_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    uint16_t *depth_mm = (uint16_t*)v_depth;
    uint32_t rainbow[] = {
        0xffff00ff, //yellow
        0x0000ffff, //blue
        0x00ff00ff, //green
        0xff0000ff, //red
        0x00ffffff, //cyan
    };
    int range = 5000;
    int step = 250;
    int n_stops = range / step;
    struct {
        float depth;
        uint32_t color;
    } stops[n_stops];

    unsigned band = 0;
    float f = 1.0f;
    for (int i = 0; i < n_stops; i++) {
        uint32_t rgba = rainbow[band];

        float r = ((rgba & 0xff000000) >> 24) / 255.0f;
        float g = ((rgba & 0xff0000) >> 16) / 255.0f;
        float b = ((rgba & 0xff00) >> 8) / 255.0f;

        r *= f;
        g *= f;
        b *= f;

        uint8_t red = (r * 255.0f) + 0.5f;
        uint8_t green = (g * 255.0f) + 0.5f;
        uint8_t blue = (b * 255.0f) + 0.5f;

        stops[i].color = ((uint32_t)red << 24) |
                         ((uint32_t)green << 16) |
                         ((uint32_t)blue << 8) |
                         0xff;
        stops[i].depth = (i * step) / 1000.0f;

        band++;
        if (band >= ARRAY_LEN(rainbow)) {
            f *= 0.8;
            band = 0;
        }
    }

    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            int in_pos = y * 640 + x;
            int out_pos = y * 640 * 3 + x * 3;
            float depth_m = depth_mm[in_pos] /  1000.0f;

            for (int i = 1; i < n_stops; i++) {

                depth_rgb_back[out_pos] = 0xff;
                depth_rgb_back[out_pos + 1] = 0xff;
                depth_rgb_back[out_pos + 2] = 0xff;
                if (depth_m < stops[i].depth) {
                    float t = (depth_m - stops[i - 1].depth) /
                        (stops[i].depth - stops[i - 1].depth);

                    uint32_t col0 = stops[i - 1].color;
                    uint32_t col1 = stops[i].color;

                    float r0 = ((col0 & 0xff000000) >> 24) / 255.0f;
                    float g0 = ((col0 & 0xff0000) >> 16) / 255.0f;
                    float b0 = ((col0 & 0xff00) >> 8) / 255.0f;

                    float r1 = ((col1 & 0xff000000) >> 24) / 255.0f;
                    float g1 = ((col1 & 0xff0000) >> 16) / 255.0f;
                    float b1 = ((col1 & 0xff00) >> 8) / 255.0f;

                    float r = (1.0f - t) * r0 + t * r1;
                    float g = (1.0f - t) * g0 + t * g1;
                    float b = (1.0f - t) * b0 + t * b1;

                    depth_rgb_back[out_pos] = (r * 255.0f) + 0.5f;
                    depth_rgb_back[out_pos + 1] = (g * 255.0f) + 0.5f;
                    depth_rgb_back[out_pos + 2] = (b * 255.0f) + 0.5f;

                    break;
                }
            }
        }
    }

    gm_context_update_depth_from_u16_mm(gm_context,
                                        0, // FIXME timestamp
                                        640, 480,
                                        (uint16_t *)v_depth);

    pthread_mutex_lock(&swap_buffers_mutex);
    std::swap(depth_rgb_back, depth_rgb_mid);
    depth_rgb_mid_valid = true;
    pthread_cond_signal(&gl_frame_cond);
    pthread_mutex_unlock(&swap_buffers_mutex);
}

static void
kinect_rgb_frame_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            int in_pos = y * 640 * 2 + x * 2;
            int out_pos = y * 640 + x;

            uint8_t lum = ((uint8_t *)rgb)[in_pos + 1];
            lum_back[out_pos] = lum;
        }
    }

    /* XXX: as far as I can tell the kinect timestamp is based off of a 60Hz
     * timer, so if we want to pass through a timestamp we should handle
     * 32bit overflow and scale into seconds/nanoseconds
     */
    gm_context_update_luminance(gm_context,
                                0, // ignored timestamp
                                640, 480,
                                lum_back);

    pthread_mutex_lock(&swap_buffers_mutex);

    assert(rgb_back == rgb);
    std::swap(rgb_back, rgb_mid);
    freenect_set_video_buffer(dev, rgb_back);

    std::swap(lum_back, lum_mid);

    rgb_mid_valid = true;

    pthread_cond_signal(&gl_frame_cond);

    pthread_mutex_unlock(&swap_buffers_mutex);
}

static void *
kinect_io_thread_cb(void *data)
{
    int state_check_throttle = 0;

    freenect_set_tilt_degs(kinect_dev, 0);
    freenect_set_led(kinect_dev, LED_RED);
    freenect_set_depth_callback(kinect_dev, kinect_depth_frame_cb);
    freenect_set_video_callback(kinect_dev, kinect_rgb_frame_cb);
    freenect_set_video_mode(kinect_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_YUV_RAW));
    //freenect_set_depth_mode(kinect_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED)); // MM, aligned to RGB
    freenect_set_depth_mode(kinect_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));
    freenect_set_video_buffer(kinect_dev, rgb_back);

    freenect_start_depth(kinect_dev);
    freenect_start_video(kinect_dev);

    while (freenect_process_events(kinect_ctx) >= 0) {
        if (state_check_throttle++ >= 2000) {
            freenect_raw_tilt_state* state;
            freenect_update_tilt_state(kinect_dev);
            state = freenect_get_tilt_state(kinect_dev);

            kinect_accel_x = state->accelerometer_x;
            kinect_accel_y = state->accelerometer_y;
            kinect_accel_z = state->accelerometer_z;

            double mks_dx, mks_dy, mks_dz;
            freenect_get_mks_accel(state, &mks_dx, &mks_dy, &mks_dz);

            kinect_mks_accel_x = mks_dx;
            kinect_mks_accel_y = mks_dy;
            kinect_mks_accel_z = mks_dz;

            kinect_tilt = freenect_get_tilt_degs(state);
            kinect_ir_brightness = freenect_get_ir_brightness(kinect_dev);

            state_check_throttle = 0;
        }
    }

    freenect_stop_depth(kinect_dev);
    freenect_stop_video(kinect_dev);

    freenect_close_device(kinect_dev);
    freenect_shutdown(kinect_ctx);

    return NULL;
}

static void
on_window_fb_size_change_cb(GLFWwindow *window, int width, int height)
{
    Data *data = (Data *)glfwGetWindowUserPointer(window);

    data->win_width = width;
    data->win_height = height;
}

static void
on_key_input_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Data *data = (Data *)glfwGetWindowUserPointer(window);

    if (action != GLFW_PRESS)
        return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(data->window, 1);
        break;
    }

    ImGui_ImplGlfwGLES3_KeyCallback(window, key, scancode, action, mods);
}

static void
on_glfw_error_cb(int error_code, const char *error_msg)
{
    fprintf(stderr, "GLFW ERROR: %d: %s\n", error_code, error_msg);
}

static void
on_khr_debug_message_cb(GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum gl_severity,
                        GLsizei length,
                        const GLchar *message,
                        void *userParam)
{
    switch (gl_severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        fprintf(stderr, "GL DEBUG: HIGH SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        fprintf(stderr, "GL DEBUG: MEDIUM SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        fprintf(stderr, "GL DEBUG: LOW SEVERITY: %s\n", message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        fprintf(stderr, "GL DEBUG: NOTIFICATION: %s\n", message);
        break;
    }
}

int
main(int argc, char **argv)
{
    Data data;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW, OpenGL windows system library\n");
        exit(1);
    }

    //depth_mid = (uint8_t*)malloc(640*480*2);
    //depth_front = (uint8_t*)malloc(640*480*2);

    depth_rgb_back = (uint8_t*)malloc(640*480*3);
    depth_rgb_mid = (uint8_t*)malloc(640*480*3);
    depth_rgb_front = (uint8_t*)malloc(640*480*3);

    rgb_back = (uint8_t*)malloc(640*480*3);
    rgb_mid = (uint8_t*)malloc(640*480*3);
    rgb_front = (uint8_t*)malloc(640*480*3);

    lum_back = (uint8_t*)malloc(640*480);
    lum_mid = (uint8_t*)malloc(640*480);
    lum_front = (uint8_t*)malloc(640*480);

    if (freenect_init(&kinect_ctx, NULL) < 0) {
        fprintf(stderr, "Failed to init libfreenect\n");
        return 1;
    }

    /* We get loads of 'errors' from the kinect but it seems to vaguely
     * be working :)
     */
    freenect_set_log_level(kinect_ctx, FREENECT_LOG_FATAL);
    freenect_select_subdevices(kinect_ctx,
                               (freenect_device_flags)(FREENECT_DEVICE_MOTOR |
                                                       FREENECT_DEVICE_CAMERA));

    if (!freenect_num_devices(kinect_ctx)) {
        fprintf(stderr, "Failed to find a Kinect device\n");
        freenect_shutdown(kinect_ctx);
        exit(1);
    }

    if (freenect_open_device(kinect_ctx, &kinect_dev, 0) < 0) {
        fprintf(stderr, "Could not open Kinect device\n");
        freenect_shutdown(kinect_ctx);
        return 1;
    }

    kinect_ir_brightness = freenect_get_ir_brightness(kinect_dev);

    freenect_raw_tilt_state *tilt_state;
    freenect_update_tilt_state(kinect_dev);
    tilt_state = freenect_get_tilt_state(kinect_dev);

    kinect_tilt = freenect_get_tilt_degs(tilt_state);

    //data.win_width = 540;
    //data.win_height = 960;
    data.win_width = 800 + TOOLBAR_LEFT_WIDTH;
    data.win_height = 600;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  0) ;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);

    data.window = glfwCreateWindow(data.win_width,
                                   data.win_height,
                                   "Glimpse Viewer", NULL, NULL);
    if (!data.window) {
        fprintf(stderr, "Failed to create window\n");
        exit(1);
    }

    glfwSetWindowUserPointer(data.window, &data);

    glfwSetFramebufferSizeCallback(data.window, on_window_fb_size_change_cb);

    glfwMakeContextCurrent(data.window);
    glfwSwapInterval(1);

    glfwSetErrorCallback(on_glfw_error_cb);


    glDebugMessageControl(GL_DONT_CARE, /* source */
                          GL_DONT_CARE, /* type */
                          GL_DONT_CARE, /* severity */
                          0,
                          NULL,
                          false);

    glDebugMessageControl(GL_DONT_CARE, /* source */
                          GL_DEBUG_TYPE_ERROR,
                          GL_DONT_CARE, /* severity */
                          0,
                          NULL,
                          true);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback((GLDEBUGPROC)on_khr_debug_message_cb, &data);

    ImGui_ImplGlfwGLES3_Init(data.window, false /* don't install callbacks */);

    /* will chain on to ImGui_ImplGlfwGLES3_KeyCallback... */
    glfwSetKeyCallback(data.window, on_key_input_cb);
    glfwSetMouseButtonCallback(data.window, ImGui_ImplGlfwGLES3_MouseButtonCallback);
    glfwSetScrollCallback(data.window, ImGui_ImplGlfwGLES3_ScrollCallback);
    glfwSetCharCallback(data.window, ImGui_ImplGlfwGLES3_CharCallback);


    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", 16.0f);

    init_opengl(&data);

    /*
     * To look more like Android we create a separate app/ui thread and a
     * separate Tango thread...
     */

    gm_context = gm_context_new(NULL);

    TangoCameraIntrinsics kinect_camera_intrinsics;
    kinect_camera_intrinsics.width = 640;
    kinect_camera_intrinsics.height = 480;

    /* libfreenect doesn't give us a way to query camera intrinsics so just
     * using these random/plausible intrinsics found on the internet to avoid
     * manually calibrating for now :)
     */
    kinect_camera_intrinsics.cx = 339.30780975300314;
    kinect_camera_intrinsics.cy = 242.73913761751615;
    kinect_camera_intrinsics.fx = 594.21434211923247;
    kinect_camera_intrinsics.fy = 591.04053696870778;

    /* Some alternative intrinsics
     *
     * TODO: we should allow explicit calibrarion and loading these at runtime
     */
#if 0
    cx = 322.515987
    cy = 259.055966
    fx = 521.179233
    fy = 493.033034
#endif

    gm_context_set_depth_camera_intrinsics(gm_context, &kinect_camera_intrinsics);

    pthread_t kinect_io_thread;
    pthread_create(&kinect_io_thread,
                   NULL, //attributes
                   kinect_io_thread_cb,
                   NULL); //data

    event_loop(&data);

    ImGui_ImplGlfwGLES3_Shutdown();
    glfwDestroyWindow(data.window);
    glfwTerminate();

    return 0;
}
