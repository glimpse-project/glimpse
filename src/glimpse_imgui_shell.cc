/*
 * Copyright (C) 2019 Glimp IP Ltd
 * Copyright (C) 2019 Robert Bragg <robert@sixbynine.org>
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <locale.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>

#include <dirent.h> // Note: we need this for PATH_MAX define on Windows
#include <limits.h>

#include <sys/stat.h>
#include <sys/types.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <list>
#include <vector>
#include <string>

#include <epoxy/gl.h>

#include <imgui.h>

#ifdef __ANDROID__
#    include <android/log.h>
#    include <jni.h>
#endif

#if defined(__APPLE__)
#    include <TargetConditionals.h>
#else
#    define TARGET_OS_MAC 0
#    define TARGET_OS_IOS 0
#    define TARGET_OS_OSX 0
#endif

#if TARGET_OS_IOS == 1
#    include "ios_utils.h"
#endif

#ifdef SUPPORTS_GLFM
#    include <glfm.h>
#    include <imgui_impl_glfm.h>
#    include <imgui_impl_opengl3.h>
#endif

#ifdef SUPPORTS_GLFW
#    include <GLFW/glfw3.h>
#    include <imgui_impl_glfw.h>
#    include <imgui_impl_opengl3.h>
#endif

#if TARGET_OS_OSX == 1 || defined(_WIN32)
#define GLSL_SHADER_VERSION "#version 400\n"
#else
#define GLSL_SHADER_VERSION "#version 300 es\n"
#endif

#include <profiler.h>

#include "glimpse_os.h"
#include "glimpse_log.h"
#include "glimpse_gl.h"
#include "glimpse_assets.h"
#include "glimpse_imgui_shell.h"
#ifdef TARGET_OS_MAC // OSX or IOS
#include "glimpse_imgui_darwin_shell.h"
#endif
#include "glimpse_imgui_shell_impl.h"

#ifdef _WIN32
#define strdup(X) _strdup(X)
#endif

#undef GM_LOG_CONTEXT
#ifdef __ANDROID__
#define GM_LOG_CONTEXT "Glimpse Shell"
#else
#define GM_LOG_CONTEXT "shell"
#endif

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))
#define LOOP_INDEX(x,y) ((x)[(y) % ARRAY_LEN(x)])

#define xsnprintf(dest, n, fmt, ...) do { \
        if (snprintf(dest, n, fmt,  __VA_ARGS__) >= (int)(n)) \
            exit(1); \
    } while(0)


#ifdef __ANDROID__
static JavaVM *android_jvm_singleton;
#endif

#ifdef SUPPORTS_GLFM
static struct gm_imgui_shell *glfm_global_shell;
#endif

static bool pause_profile;

extern void
glimpse_imgui_shell_main(struct gm_imgui_shell *shell,
                         int argc,
                         char **argv);

static void
on_khr_debug_message_cb(GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum gl_severity,
                        GLsizei length,
                        const GLchar *message,
                        void *user_data)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)user_data;

    switch (gl_severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        gm_log(shell->log, GM_LOG_ERROR, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        gm_log(shell->log, GM_LOG_WARN, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        gm_log(shell->log, GM_LOG_WARN, "Viewer GL", "%s", message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        gm_log(shell->log, GM_LOG_INFO, "Viewer GL", "%s", message);
        break;
    }
}

static void
opengl_init(struct gm_imgui_shell *shell)
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClearStencil(0);

    if (epoxy_has_gl_extension("GL_KHR_debug")) {
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
        glDebugMessageCallback((GLDEBUGPROC)on_khr_debug_message_cb, shell);
    }

    if (!shell->gl_is_embedded) {
        // In the forwards-compatible context, there's no default vertex array.
        GLuint vertex_array;
        glGenVertexArrays(1, &vertex_array);
        glBindVertexArray(vertex_array);
    }

    GM_GL_CHECK_ERRORS(shell->log);

    shell->gl_initialized = true;
}

static void
on_profiler_pause_cb(bool pause)
{
    pause_profile = pause;
}

static void
imgui_init(struct gm_imgui_shell *shell)
{
    gm_info(shell->log, "Initializing IMGUI state...");

    ImGui::StyleColorsClassic();

    // We don't try and load any external fonts since we might not have
    // permission to access any assets. We leave it up to specific
    // applications to load whatever fonts they want (possibly after
    // checking for permissions)

    ProfileInitialize(&pause_profile, on_profiler_pause_cb);

    shell->imgui_initialized = true;
}


#ifdef SUPPORTS_GLFM
static void
glfm_surface_created_cb(GLFMDisplay *display, int width, int height)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfmGetUserData(display);

    gm_debug(shell->log, "Surface created (%dx%d)", width, height);

    shell->surface_width = width;
    shell->surface_height = height;

    if (shell->surface_created_callback) {
        shell->surface_created_callback(shell, width, height,
                                        shell->surface_created_callback_data);
    }
}

static void
glfm_surface_resized_cb(GLFMDisplay *display, int width, int height)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfmGetUserData(display);

    gm_debug(shell->log, "Surface resized (%dx%d)", width, height);

    if (shell->surface_resized_callback) {
        shell->surface_resized_callback(shell, width, height,
                                        shell->surface_resized_callback_data);
    }

}

static void
glfm_surface_destroyed_cb(GLFMDisplay *display)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfmGetUserData(display);

    gm_debug(shell->log, "Surface destroyed");
}

static void
glfm_app_focus_cb(GLFMDisplay *display, bool focused)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfmGetUserData(display);

    gm_debug(shell->log, focused ? "Focused" : "Unfocused");
}

static void
glfm_mainloop_cb(GLFMDisplay* display, double frameTime)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfmGetUserData(display);

    uint64_t time = frameTime * 1e9;

    if (!shell->imgui_initialized) {
        imgui_init(shell);
    }

    ProfileNewFrame();
    ProfileScopedSection(Frame);

    if (shell->mainloop_callback)
    {
        ProfileScopedSection(MainAppLogic);

        shell->mainloop_callback(shell,
                                 time,
                                 shell->mainloop_callback_data);
    }


    {
        ProfileScopedSection(Redraw);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfm_NewFrame(display, frameTime);
        ImGui::NewFrame();

        if (!shell->gl_initialized)
            opengl_init(shell);

        glViewport(0, 0, shell->surface_width, shell->surface_height);
        glClear(GL_COLOR_BUFFER_BIT);

        if (shell->render_callback)
        {
            ProfileScopedSection(AppRenderLogic);

            shell->render_callback(shell,
                                   time,
                                   shell->render_callback_data);
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

static void
glfm_init(struct gm_imgui_shell *shell)
{
    GLFMDisplay *display = shell->glfm_display;

    glfmSetDisplayConfig(display,
                         GLFMRenderingAPIOpenGLES3,
                         GLFMColorFormatRGBA8888,
                         GLFMDepthFormatNone,
                         GLFMStencilFormatNone,
                         GLFMMultisampleNone);
    glfmSetDisplayChrome(display,
                         GLFMUserInterfaceChromeNavigationAndStatusBar);
    glfmSetUserData(display, shell);
    glfmSetSurfaceCreatedFunc(display, glfm_surface_created_cb);
    glfmSetSurfaceResizedFunc(display, glfm_surface_resized_cb);
    glfmSetSurfaceDestroyedFunc(display, glfm_surface_destroyed_cb);
    glfmSetAppFocusFunc(display, glfm_app_focus_cb);
    glfmSetMainLoopFunc(display, glfm_mainloop_cb);

    shell->gl_is_embedded = true;

    ImGui::CreateContext();
    ImGui_ImplGlfm_Init(display, true /* install callbacks */);
    ImGui_ImplOpenGL3_Init(GLSL_SHADER_VERSION);

    // Quick hack to make scrollbars a bit more usable on small devices
    ImGui::GetStyle().ScrollbarSize *= 2;
}
#endif // SUPPORTS_GLFM

#ifdef SUPPORTS_GLFW
static void
glfw_mainloop(struct gm_imgui_shell *shell)
{
    gm_info(shell->log, "Starting GLFW mainloop loop...");

    while (!glfwWindowShouldClose(shell->glfw_window)) {
        uint64_t time = gm_os_get_time();

        ProfileNewFrame();

        ProfileScopedSection(Frame);

        {
            ProfileScopedSection(GLFWEvents);
            glfwPollEvents();
        }

        if (shell->mainloop_callback)
        {
            ProfileScopedSection(MainAppLogic);

            shell->mainloop_callback(shell,
                                     time,
                                     shell->mainloop_callback_data);
        }

        {
            ProfileScopedSection(Redraw);

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            glfwMakeContextCurrent(shell->glfw_window);

            if (!shell->gl_initialized)
                opengl_init(shell);

            glViewport(0, 0, shell->surface_width, shell->surface_height);
            glClear(GL_COLOR_BUFFER_BIT);

            if (shell->render_callback)
            {
                ProfileScopedSection(AppRenderLogic);

                shell->render_callback(shell,
                                       time,
                                       shell->render_callback_data);
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        {
            ProfileScopedSection(SwapBuffers);
            glfwMakeContextCurrent(shell->glfw_window);
            glfwSwapBuffers(shell->glfw_window);
        }
    }
}

static void
glfw_window_fb_size_change_cb(GLFWwindow *window, int width, int height)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfwGetWindowUserPointer(window);

    shell->surface_width = width;
    shell->surface_height = height;

    if (shell->surface_resized_callback) {
        shell->surface_resized_callback(shell, width, height,
                                        shell->surface_resized_callback_data);
    }
}

static void
glfw_key_input_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    struct gm_imgui_shell *shell =
        (struct gm_imgui_shell *)glfwGetWindowUserPointer(window);

    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

    if (action != GLFW_PRESS)
        return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(shell->glfw_window, 1);
        break;
    }
}

static void
glfw_error_cb(int error_code, const char *error_msg)
{
    fprintf(stderr, "GLFW ERROR: %d: %s\n", error_code, error_msg);
}

static bool
glfw_init(struct gm_imgui_shell *shell, char **err)
{
    if (!glfwInit()) {
        gm_throw(shell->log, err,
                 "Failed to init GLFW, OpenGL windows system library\n");
        return false;
    }

#if TARGET_OS_OSX == 1 || defined(_WIN32) || defined(__linux__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  2) ;
    shell->gl_is_embedded = false;
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3) ;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,  0) ;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    shell->gl_is_embedded = true;
#endif

    shell->surface_width = 1280;
    shell->surface_height = 720;

    shell->glfw_window = glfwCreateWindow(shell->surface_width,
                                     shell->surface_height,
                                     shell->app_title,
                                     NULL, NULL);
    if (!shell->glfw_window) {
        gm_throw(shell->log, err, "Failed to create window\n");
        return false;
    }

    glfwSetWindowUserPointer(shell->glfw_window, shell);

    glfwGetFramebufferSize(shell->glfw_window,
                           &shell->surface_width,
                           &shell->surface_height);
    glfwSetFramebufferSizeCallback(shell->glfw_window, glfw_window_fb_size_change_cb);

    glfwMakeContextCurrent(shell->glfw_window);
    glfwSwapInterval(1);

    glfwSetErrorCallback(glfw_error_cb);

    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(shell->glfw_window, false /* don't install callbacks */);
    ImGui_ImplOpenGL3_Init(GLSL_SHADER_VERSION);

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 ui_scale = io.DisplayFramebufferScale;
    ImGui::GetStyle().ScaleAllSizes(ui_scale.x);

    /* will chain on to ImGui_ImplGlfw_KeyCallback... */
    glfwSetKeyCallback(shell->glfw_window, glfw_key_input_cb);
    glfwSetMouseButtonCallback(shell->glfw_window,
                               ImGui_ImplGlfw_MouseButtonCallback);
    glfwSetScrollCallback(shell->glfw_window, ImGui_ImplGlfw_ScrollCallback);
    glfwSetCharCallback(shell->glfw_window, ImGui_ImplGlfw_CharCallback);

    if (shell->surface_created_callback) {
        shell->surface_created_callback(shell,
                                        shell->surface_width,
                                        shell->surface_height,
                                        shell->surface_created_callback_data);
    }

    imgui_init(shell);

    return true;
}
#endif // SUPPORTS_GLFW

static void
logger_cb(struct gm_logger *logger,
          enum gm_log_level level,
          const char *context,
          struct gm_backtrace *backtrace,
          const char *format,
          va_list ap,
          void *user_data)
{
    struct gm_imgui_shell *shell = (struct gm_imgui_shell *)user_data;
    char *msg = NULL;

    xvasprintf(&msg, format, ap);

#ifdef __ANDROID__
    switch (level) {
    case GM_LOG_ASSERT:
        __android_log_print(ANDROID_LOG_FATAL, context, "%s", msg);
        break;
    case GM_LOG_ERROR:
        __android_log_print(ANDROID_LOG_ERROR, context, "%s", msg);
        break;
    case GM_LOG_WARN:
        __android_log_print(ANDROID_LOG_WARN, context, "%s", msg);
        break;
    case GM_LOG_INFO:
        __android_log_print(ANDROID_LOG_INFO, context, "%s", msg);
        break;
    case GM_LOG_DEBUG:
        __android_log_print(ANDROID_LOG_DEBUG, context, "%s", msg);
        break;
    }
#endif

    if (shell->log_fp) {
        switch (level) {
        case GM_LOG_ERROR:
            fprintf(shell->log_fp, "%s: ERROR: ", context);
            break;
        case GM_LOG_WARN:
            fprintf(shell->log_fp, "%s: WARN: ", context);
            break;
        default:
            fprintf(shell->log_fp, "%s: ", context);
        }

        fprintf(shell->log_fp, "%s\n", msg);
#if TARGET_OS_IOS == 1
        ios_log(msg);
#endif

        if (backtrace) {
            int line_len = 100;
            char *formatted = (char *)alloca(backtrace->n_frames * line_len);

            gm_logger_get_backtrace_strings(logger, backtrace,
                                            line_len, (char *)formatted);
            for (int i = 0; i < backtrace->n_frames; i++) {
                char *line = formatted + line_len * i;
                fprintf(shell->log_fp, "> %s\n", line);
            }
        }

        fflush(shell->log_fp);
        fflush(stdout);
    }

    xfree(msg);
}

static void
logger_abort_cb(struct gm_logger *logger,
                void *user_data)
{
    struct gm_imgui_shell *shell = (struct gm_imgui_shell *)user_data;

    if (shell->log_fp) {
        fprintf(shell->log_fp, "ABORT\n");
        fflush(shell->log_fp);
        fclose(shell->log_fp);
    }

    abort();
}

void
gm_imgui_shell_preinit_log(struct gm_imgui_shell *shell,
                           struct gm_logger *log)
{
    if (shell->initialized) {
        gm_error(shell->log, "_preinit apis must be called before gm_imgui_shell_init()");
        gm_logger_abort(shell->log);
    }

    shell->log = log;
}

void
gm_imgui_shell_preinit_log_filename(struct gm_imgui_shell *shell,
                                    const char *log_filename)
{
    if (shell->initialized) {
        gm_error(shell->log, "_preinit apis must be called before gm_imgui_shell_init()");
        gm_logger_abort(shell->log);
    }

    shell->log_filename = strdup(log_filename);
}

void
gm_imgui_shell_preinit_assets_root(struct gm_imgui_shell *shell,
                                   const char *assets_root)
{
    if (shell->initialized) {
        gm_error(shell->log, "_preinit apis must be called before gm_imgui_shell_init()");
        gm_logger_abort(shell->log);
    }

    if (shell->custom_assets_root) {
        free(shell->custom_assets_root);
        shell->custom_assets_root = NULL;
    }

    shell->custom_assets_root = assets_root ? strdup(assets_root) : NULL;
}

bool
gm_imgui_shell_preinit_renderer(struct gm_imgui_shell *shell,
                                enum gm_imgui_renderer renderer)
{
    if (shell->initialized) {
        gm_error(shell->log, "_preinit apis must be called before gm_imgui_shell_init()");
        gm_logger_abort(shell->log);
    }

#ifndef SUPPORTS_METAL
    if (renderer == GM_IMGUI_RENDERER_METAL)
    {
        gm_info(shell->log, "Probe failure for Meta API support");
        return false;
    }
#endif

    shell->requested_renderer = renderer;

    return true;
}

#define IMPL_CALLBACK_PREINIT_API(CALLBACK_NAME, CALLBACK_TYPE) \
void \
gm_imgui_shell_preinit_##CALLBACK_NAME##_callback(struct gm_imgui_shell *shell, \
                                                  CALLBACK_TYPE, \
                                                  void *user_data) \
{ \
    if (shell->initialized) { \
        gm_error(shell->log, \
                 "_preinit apis must be called before gm_imgui_shell_init()"); \
        gm_logger_abort(shell->log); \
    } \
 \
    shell->CALLBACK_NAME##_callback = callback; \
    shell->CALLBACK_NAME##_callback_data = user_data; \
}
IMPL_CALLBACK_PREINIT_API(log_ready,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           struct gm_logger *log,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(surface_created,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           int width,
                                           int height,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(surface_resized,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           int width,
                                           int height,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(surface_destroyed,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(app_focus,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           bool focused,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(mainloop,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           uint64_t timestamp,
                                           void *user_data))
IMPL_CALLBACK_PREINIT_API(render,
                          void (*callback)(struct gm_imgui_shell *shell,
                                           uint64_t timestamp,
                                           void *user_data))
#undef IMPL_CALLBACK_PREINIT_API


static void __attribute__((unused))
imgui_shell_destroy(struct gm_imgui_shell *shell)
{
#ifdef SUPPORTS_GLFW
    if (shell->surface_destroyed_callback) {
        shell->surface_destroyed_callback(shell,
                                          shell->surface_created_callback_data);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // defer destroying the window to ensure the GL context can remain bound
    // to something since the imgui teardown will call into GL.
    glfwDestroyWindow(shell->glfw_window);
    glfwTerminate();
#endif

    ProfileShutdown();

    // if log_fp is NULL then we don't own the logger (it was specified
    // via _preinit_log()
    if (shell->log_fp) {
        gm_logger_destroy(shell->log);
    }

    if (shell->log_filename)
        free(shell->log_filename);

    delete shell;
}

bool
gm_imgui_shell_init(struct gm_imgui_shell *shell,
                    const char *app_name,
                    const char *app_title,
                    char **err)
{
    if (shell->initialized) {
        gm_throw(shell->log, err, "Can't re-initialize shell");
        return false;
    }

    if (shell->app_name) {
        free(shell->app_name);
        shell->app_name = NULL;
    }
    shell->app_name = strdup(app_name);

    if (shell->app_title) {
        free(shell->app_title);
        shell->app_title = NULL;
    }
    shell->app_title = strdup(app_title);

    char *assets_root = NULL;

    if (shell->custom_assets_root) {
        assets_root = strdup(shell->custom_assets_root);
    } else {
#if TARGET_OS_IOS == 1
        assets_root = ios_util_get_documents_path();
#elif defined(__ANDROID__)
        char assets_root_tmp[PATH_MAX];
        snprintf(assets_root_tmp, sizeof(assets_root_tmp),
                 "/sdcard/%s", app_name);
        assets_root = strdup(assets_root_tmp);
#else

        const char *assets_root_env = getenv("GLIMPSE_ASSETS_ROOT");
        assets_root = strdup(assets_root_env ? assets_root_env : "");
        if (assets_root_env) {
            fprintf(stderr, "GLIMPSE_ASSETS_ROOT=%s\n", assets_root_env);
        }
#endif
    }

    if (!shell->log) {
        char log_filename_tmp[PATH_MAX];
        const char *log_filename_debug = NULL;

        if (shell->log_filename) {
            log_filename_debug = shell->log_filename;
            shell->log_fp = fopen(shell->log_filename, "w");
        } else {
#if TARGET_OS_IOS == 1
            snprintf(log_filename_tmp, sizeof(log_filename_tmp),
                     "%s/glimpse.log", assets_root);
            log_filename_debug = log_filename_tmp;
            shell->log_fp = fopen(log_filename_tmp, "w");
#elif defined(__ANDROID__)
            snprintf(log_filename_tmp, sizeof(log_filename_tmp),
                     "/sdcard/%s/glimpse.log", app_name);
            log_filename_debug = log_filename_tmp;
            shell->log_fp = fopen(log_filename_tmp, "w");
#else
            log_filename_debug = "stderr";
            shell->log_fp = stderr;
#endif
        }

        if (!shell->log_fp) {
            if (!err) {
                fprintf(stderr, "Failed to open a log file (%s)",
                        log_filename_debug);
                fflush(stderr);
                abort();
            }
            xasprintf(err, "Failed to open a log file (%s)",
                      log_filename_debug);
            return false;
        }

        shell->log = gm_logger_new(logger_cb, shell);
        gm_logger_set_abort_callback(shell->log, logger_abort_cb, shell);

        gm_info(shell->log, "Opened log file %s", log_filename_debug);
    }

    gm_debug(shell->log, "Glimpse Shell");

    if (shell->log_ready_callback) {
        shell->log_ready_callback(shell, shell->log,
                                  shell->log_ready_callback_data);
    }

    gm_set_assets_root(shell->log, assets_root);


    /* Our subproject copy of libfreenect doesn't have this issue but upstream
     * fakenect may forcibly exit an application if FAKENECT_PATH is not set in
     * the environment so we try and avoid that...
     */
#ifdef USE_FREENECT
    if (!getenv("FAKENECT_PATH")) {

        char fakenect_path[14 + PATH_MAX];
        struct stat sb;

        snprintf(fakenect_path, sizeof(fakenect_path),
                 "%s/FakeRecording", assets_root);
        gm_warn(shell->log,
                "Automatically setting FAKENECT_PATH=%s to avoid exit() by fakenect",
                fakenect_path);

        if (stat(fakenect_path, &sb) != -1 && S_ISDIR(sb.st_mode))
        {
#ifdef _WIN32
            _putenv_s("FAKENECT_PATH", fakenect_path);
#else
            setenv("FAKENECT_PATH", fakenect_path, true);
#endif
        }
    }
#endif

    free(assets_root);
    assets_root = NULL;

    const char *renderer_override = getenv("GLIMPSE_RENDERER");
    if (renderer_override)
    {
        if (strcasecmp(renderer_override, "GL") == 0)
            shell->requested_renderer = GM_IMGUI_RENDERER_OPENGL;
        else if (strcasecmp(renderer_override, "metal") == 0)
            shell->requested_renderer = GM_IMGUI_RENDERER_METAL;
    }

    switch (shell->requested_renderer)
    {
    case GM_IMGUI_RENDERER_AUTO:
    case GM_IMGUI_RENDERER_OPENGL:
        shell->renderer = GM_IMGUI_RENDERER_OPENGL;
        break;
    case GM_IMGUI_RENDERER_METAL:
        shell->renderer = GM_IMGUI_RENDERER_METAL;
        break;
    }

    switch (shell->renderer)
    {
    case GM_IMGUI_RENDERER_METAL:
        shell->winsys = GM_IMGUI_WINSYS_METAL_KIT;
        break;
    case GM_IMGUI_RENDERER_OPENGL:
#ifdef SUPPORTS_GLFM
        shell->winsys = GM_IMGUI_WINSYS_GLFM;
#endif
#ifdef SUPPORTS_GLFW
        shell->winsys = GM_IMGUI_WINSYS_GLFW;
#endif
        break;
    case GM_IMGUI_RENDERER_AUTO:
        gm_assert(shell->log, 0, "Spurious undefined shell renderer");
        break;
    }

    // We can't init GLFM at this point because we have to
    // start the mainloop before glfmMain will be called
    // on some platforms

#ifdef SUPPORTS_GLFW
    if (shell->winsys == GM_IMGUI_WINSYS_GLFW)
    {
        gm_info(shell->log, "Initializing GLFW...");
        if (!glfw_init(shell, err)) {
            return false;
        }
    }
#endif

    shell->initialized = true;

    return true;
}

struct gm_logger *
gm_imgui_shell_get_log(struct gm_imgui_shell *shell)
{
    return shell->log;
}

enum gm_imgui_renderer
gm_imgui_shell_get_renderer(struct gm_imgui_shell *shell)
{
    if (shell->renderer == GM_IMGUI_RENDERER_AUTO) {
        gm_error(shell->log, "%s called before the renderer has been determined", __func__);
        gm_logger_abort(shell->log);
    }

    return shell->renderer;
}

bool
gm_imgui_shell_is_gles_renderer(struct gm_imgui_shell *shell)
{
    return shell->gl_is_embedded;
}

void
gm_imgui_shell_get_chrome_insets(struct gm_imgui_shell *shell,
                                 float *top,
                                 float *right,
                                 float *bottom,
                                 float *left)
{
    *top = 0;
    *right = 0;
    *bottom = 0;
    *left = 0;

#ifdef SUPPORTS_GLFM
    double t, r, b, l;
    glfmGetDisplayChromeInsets(shell->glfm_display, &t, &r, &b, &l);
    *top = t;
    *right = r;
    *bottom = b;
    *left = l;
#endif
}

#ifdef SUPPORTS_GLFM
void
glfmMain(GLFMDisplay *display)
{
    struct gm_imgui_shell *shell = glfm_global_shell;

    // In the case of Android then glfmMain is our first entry point
    if (!shell)
    {
        shell = new gm_imgui_shell();
        shell->renderer = GM_IMGUI_RENDERER_OPENGL;
        glimpse_imgui_shell_main(shell, 0, NULL);
    }

    shell->glfm_display = display;

    gm_info(shell->log, "Initializing GLFM...");
    glfm_init(shell);
}
#endif // SUPPORTS_GLFM

//static void
//darwin_metal_kit_app_ready_cb(struct gm_imgui_shell *shell)
//{
//    gm_info(shell->log, "Darwin Metal Kit app ready");
//}

#ifndef __ANDROID__
int
main(int argc, char **argv)
{
    struct gm_imgui_shell *shell = new gm_imgui_shell();

    // This isn't assuming we will use GLFM, but if we do
    // then glfmMain will be called later and needs to be
    // able to access this shell...
#ifdef SUPPORTS_GLFM
    glfm_global_shell = shell;
#endif

    glimpse_imgui_shell_main(shell, argc, argv);

    switch (shell->winsys)
    {
    case GM_IMGUI_WINSYS_METAL_KIT:
#ifdef SUPPORTS_METAL
        glimpse_imgui_darwin_metal_kit_main(shell);
#endif
        break;
    case GM_IMGUI_WINSYS_GLFM:
#ifdef SUPPORTS_GLFM
#if TARGET_OS_IOS == 1
        glimpse_imgui_darwin_glfm_main(shell);
#else
#error "Unhandled main()-based GLFM init"
#endif
#endif
        break;
    case GM_IMGUI_WINSYS_GLFW:
#ifdef SUPPORTS_GLFW
        glfw_mainloop(shell);
#endif
        break;
    case GM_IMGUI_WINSYS_NONE:
        gm_assert(shell->log, 0, "Spurious, undefined, shell winsys");
        break;
    }

    imgui_shell_destroy(shell);

    return 0;
}
#endif // !ANDROID


#ifdef __ANDROID__
extern "C" jint
JNI_OnLoad(JavaVM *vm, void *reserved)
{
    android_jvm_singleton = vm;

    return JNI_VERSION_1_6;
}
#endif
