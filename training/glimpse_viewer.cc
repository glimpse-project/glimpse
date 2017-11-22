#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <locale.h>
#include <assert.h>
#include <time.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <math.h>

#include <pthread.h>

#include <vector>

#include <epoxy/gl.h>
#include <epoxy/egl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "glimpse_context.h"
#include "glimpse_device.h"
#include "half.hpp"

#include <imgui.h>
#include <imgui_impl_glfw_gles3.h>
#include <profiler.h>

#define TOOLBAR_LEFT_WIDTH 300

using half_float::half;

enum event_type
{
    EVENT_DEVICE,
    EVENT_CONTEXT
};

struct event
{
    enum event_type type;
    union {
        struct gm_event *context_event;
        struct gm_device_event *device_event;
    };
};

typedef struct _Data
{
    struct gm_context *ctx;
    struct gm_device *device;

    GLFWwindow *window;
    int win_width;
    int win_height;

    int attr_pos;
    int attr_tex_coord;
    int attr_color;

    /* A convenience for accessing the depth_camera_intrinsics.width/height */
    int depth_width;
    int depth_height;

    /* A convenience for accessing the video_camera_intrinsics.width/height */
    int video_width;
    int video_height;

    /* When we request gm_device for a frame we set requirements for what the
     * frame should include. We track the requirements so we avoid sending
     * subsequent frame requests that would downgrade the requirements
     */
    uint64_t pending_frame_requirements;

    /* Set when gm_device sends a _FRAME_READY device event */
    bool device_frame_ready;

    /* Once we've been notified that there's a device frame ready for us then
     * we store the latest frame from gm_device_get_latest_frame() here...
     *
     * NB: this frame is only valid to access up until the next call to
     * gm_device_get_latest_frame() because the gm_device api is free to
     * recycle the back buffers that are part of a frame.
     */
    struct gm_frame *device_frame;

    /* Set when gm_context sends a _REQUEST_FRAME event */
    bool context_needs_frame;
    /* Set when gm_context sends a _TRACKING_READY event */
    bool tracking_ready;

    /* Once we've been notified that there's a skeleton tracking update for us
     * then we store the latest tracking data from
     * gm_context_get_latest_tracking() here...
     */
    struct gm_tracking *latest_tracking;

    /* Events from the gm_context and gm_device apis may be delivered via any
     * arbitrary thread which we don't want to block, and at a time where
     * the gm_ apis may not be reentrant due to locks held during event
     * notification
     */
    pthread_mutex_t event_queue_lock;
    std::vector<struct event> *events_back;
    std::vector<struct event> *events_front;
} Data;

static GLuint gl_tex_program;
static GLuint uniform_tex_sampler;
static GLuint gl_labels_tex;
static GLuint gl_depth_rgb_tex;
static GLuint gl_rgb_tex;
static GLuint gl_lum_tex;

static bool pause_profile;

static void
on_profiler_pause_cb(bool pause)
{
    pause_profile = pause;
}

static void
draw_ui(Data *data)
{
    float left_col = TOOLBAR_LEFT_WIDTH;
    ImVec2 main_menu_size;
    ImVec2 win_size;
    ProfileScopedSection(DrawIMGUI, ImGuiControl::Profiler::Dark);

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

    struct gm_ui_properties *props = gm_context_get_ui_properties(data->ctx);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        if (prop->type == GM_PROPERTY_INT)
            ImGui::SliderInt(prop->name, prop->int_ptr, prop->min, prop->max);
        if (prop->type == GM_PROPERTY_FLOAT)
            ImGui::SliderFloat(prop->name, prop->float_ptr, prop->min, prop->max);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    props = gm_device_get_ui_properties(data->device);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];

        switch (prop->type) {
        case GM_PROPERTY_INT:
            ImGui::SliderInt(prop->name, prop->int_ptr, prop->min, prop->max);
            break;
        case GM_PROPERTY_FLOAT:
            ImGui::SliderFloat(prop->name, prop->float_ptr, prop->min, prop->max);
            break;
        case GM_PROPERTY_FLOAT_VEC3:
            if (prop->read_only) {
                ImGui::LabelText(prop->name, "%.3f,%.3f,%.3f",
                                 prop->float_vec3[0],
                                 prop->float_vec3[1],
                                 prop->float_vec3[2]);
            } // else TODO
            break;
        }
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

    ProfileDrawUI();

    ImGui::Render();
}

static void
redraw(Data *data)
{
    ProfileScopedSection(Redraw);

    glViewport(0, 0, data->win_width, data->win_height);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_ui(data);
}

/* If we've already requested gm_device for a frame then this won't submit
 * a request that downgrades the requirements
 */
static void
request_device_frame(Data *data, uint64_t requirements)
{
    uint64_t new_requirements = data->pending_frame_requirements | requirements;

    if (data->pending_frame_requirements != new_requirements) {
        gm_device_request_frame(data->device, new_requirements);
        data->pending_frame_requirements = new_requirements;
    }
}

static void
handle_device_frame_updates(Data *data)
{
    ProfileScopedSection(UpdatingDeviceFrame);
    bool upload = false;

    if (!data->device_frame_ready)
        return;

    /* XXX We have to consider that a gm_frame currently only remains valid
     * to access until the next call to gm_device_get_latest_frame().
     *
     * Conceptually we have two decoupled consumers: 1) this redraw/render
     * loop 2) skeletal tracking so we need to be careful about
     * understanding the required gm_frame lifetime.
     *
     * Since we can currently assume gm_context_notify_frame() will
     * internally copy whatever frame data it requires then so long as we
     * synchronize these calls with the redraw loop we know it's safe to
     * free the last gm_frame once we have received a new one.
     */

    if (data->device_frame) {
        ProfileScopedSection(FreeFrame);
        gm_device_free_frame(data->device, data->device_frame);
    }

    {
        ProfileScopedSection(GetLatestFrame);
        data->device_frame = gm_device_get_latest_frame(data->device);
        assert(data->device_frame);
        upload = true;
    }

    if (data->context_needs_frame) {
        ProfileScopedSection(FwdContextFrame);

        data->context_needs_frame =
            !gm_context_notify_frame(data->ctx, data->device_frame);
    }

    data->device_frame_ready = false;

    {
        ProfileScopedSection(DeviceFrameRequest);

        /* immediately request a new frame since we want to render the camera
         * at the native capture rate, even though we might not be tracking
         * at that rate.
         *
         * Note: the requirements may be upgraded to ask for _DEPTH data
         * after the next iteration of skeltal tracking completes.
         */
        request_device_frame(data, GM_REQUEST_FRAME_LUMINANCE);
    }

    if (upload) {
        ProfileScopedSection(UploadFrameTextures);

        /*
         * Update luminance from RGB camera
         */
        glBindTexture(GL_TEXTURE_2D, gl_lum_tex);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        /* NB: gles2 only allows npot textures with clamp to edge
         * coordinate wrapping
         */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        void *video_front = gm_frame_get_video_buffer(data->device_frame);
        enum gm_format video_format = gm_frame_get_video_format(data->device_frame);

        assert(video_format == GM_FORMAT_LUMINANCE_U8);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                     data->video_width, data->video_height,
                     0, GL_LUMINANCE, GL_UNSIGNED_BYTE, video_front);
    }
}

static void
upload_tracking_textures(Data *data)
{
    ProfileScopedSection(UploadTrackingBufs);

    /*
     * Update the RGB visualization of the depth buffer
     */
    glBindTexture(GL_TEXTURE_2D, gl_depth_rgb_tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    /* NB: gles2 only allows npot textures with clamp to edge
     * coordinate wrapping
     */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    const uint8_t *depth_rgb = gm_tracking_get_rgb_depth(data->latest_tracking);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 data->depth_width, data->depth_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, depth_rgb);

    /*
     * Update inferred label map
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

    const uint8_t *labels_rgb =
        gm_tracking_get_rgb_label_map(data->latest_tracking,
                                      &label_map_width,
                                      &label_map_height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 label_map_width, label_map_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, labels_rgb);
}

static void
handle_context_tracking_updates(Data *data)
{
    ProfileScopedSection(UpdatingTracking);

    if (!data->tracking_ready)
        return;

    data->tracking_ready = false;
    data->latest_tracking = gm_context_get_latest_tracking(data->ctx);
    assert(data->latest_tracking);
    
    upload_tracking_textures(data);
}

static void
handle_device_event(Data *data, struct gm_device_event *event)
{
    switch (event->type) {
    case GM_DEV_EVENT_FRAME_READY:

        /* It's always possible that we will see an event for a frame
         * that was ready before we upgraded the requirements for what
         * we need, so we skip notifications for frames we can't use.
         */
        if (event->frame_ready.met_requirements ==
            data->pending_frame_requirements)
        {
            data->pending_frame_requirements = 0;
            data->device_frame_ready = true;
        }
        break;
    }

    gm_device_event_free(event);
}

static void
handle_context_event(Data *data, struct gm_event *event)
{
    switch (event->type) {
    case GM_EVENT_REQUEST_FRAME:
        data->context_needs_frame = true;
        request_device_frame(data,
                             (GM_REQUEST_FRAME_DEPTH |
                              GM_REQUEST_FRAME_LUMINANCE));
        break;
    case GM_EVENT_TRACKING_READY:
        data->tracking_ready = true;
        break;
    }

    gm_context_event_free(event);
}

static void
event_loop(Data *data)
{
    while (!glfwWindowShouldClose(data->window)) {
        ProfileNewFrame();

        ProfileScopedSection(Frame);

        {
            ProfileScopedSection(GLFWEvents);
            glfwPollEvents();
        }

        {
            ProfileScopedSection(GlimpseEvents);

            pthread_mutex_lock(&data->event_queue_lock);
            std::swap(data->events_front, data->events_back);
            pthread_mutex_unlock(&data->event_queue_lock);

            for (unsigned i = 0; i < data->events_front->size(); i++) {
                struct event event = (*data->events_front)[i];

                switch (event.type) {
                case EVENT_DEVICE:
                    handle_device_event(data, event.device_event);
                    break;
                case EVENT_CONTEXT:
                    handle_context_event(data, event.context_event);
                    break;
                }
            }

            data->events_front->clear();
        }

        handle_device_frame_updates(data);
        handle_context_tracking_updates(data);

        {
            ProfileScopedSection(GlimpseGPUHook);
            gm_context_render_thread_hook(data->ctx);
        }

        redraw(data);

        {
            ProfileScopedSection(SwapBuffers);
            glfwSwapBuffers(data->window);
        }
    }
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

/* NB: it's undefined what thread this is called on and we are currently
 * assuming it's safe to call gm_device_request_frame() from any thread
 * considering that it just sets a bitmask and signals a condition variable.
 */
static void
on_event_cb(struct gm_context *ctx,
            struct gm_event *context_event, void *user_data)
{
    Data *data = (Data *)user_data;

    struct event event = {};
    event.type = EVENT_CONTEXT;
    event.context_event = context_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
}

static void
on_device_event_cb(struct gm_device *dev,
                   struct gm_device_event *device_event,
                   void *user_data)
{
    Data *data = (Data *)user_data;

    struct event event = {};
    event.type = EVENT_DEVICE;
    event.device_event = device_event;

    pthread_mutex_lock(&data->event_queue_lock);
    data->events_back->push_back(event);
    pthread_mutex_unlock(&data->event_queue_lock);
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

int
main(int argc, char **argv)
{
    Data data = {};

    data.events_front = new std::vector<struct event>();
    data.events_back = new std::vector<struct event>();

    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW, OpenGL windows system library\n");
        exit(1);
    }

    if (argc == 2) {
        struct gm_device_config config = {};
        config.type = GM_DEVICE_RECORDING;
        config.recording.path = argv[1];
        data.device = gm_device_open(&config, NULL);
    } else {
        struct gm_device_config config = {};
        config.type = GM_DEVICE_KINECT;
        data.device = gm_device_open(&config, NULL);
    }

    struct gm_intrinsics *depth_intrinsics =
        gm_device_get_depth_intrinsics(data.device);
    data.depth_width = depth_intrinsics->width;
    data.depth_height = depth_intrinsics->height;

    struct gm_intrinsics *video_intrinsics =
        gm_device_get_video_intrinsics(data.device);
    data.video_width = video_intrinsics->width;
    data.video_height = video_intrinsics->height;

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

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    init_opengl(&data);

    data.ctx = gm_context_new(NULL);

    gm_context_set_depth_camera_intrinsics(data.ctx, depth_intrinsics);
    gm_context_set_video_camera_intrinsics(data.ctx, video_intrinsics);

    /* NB: there's no guarantee about what thread these event callbacks
     * might be invoked from...
     */
    gm_context_set_event_callback(data.ctx, on_event_cb, &data);
    gm_device_set_event_callback(data.device, on_device_event_cb, &data);

    gm_device_start(data.device);
    gm_context_enable(data.ctx);

    event_loop(&data);

    gm_context_destroy(data.ctx);

    ProfileShutdown();
    ImGui_ImplGlfwGLES3_Shutdown();
    glfwDestroyWindow(data.window);
    glfwTerminate();

    gm_device_close(data.device);

    delete data.events_front;
    delete data.events_back;

    return 0;
}
