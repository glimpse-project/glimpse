/*
 * Copyright (C) 2019 Glimp IP Ltd
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


#pragma once

#include <stdio.h>

#ifdef SUPPORTS_GLFM
#include <glfm.h>
#endif

#ifdef SUPPORTS_GLFW
#include <GLFW/glfw3.h>
#endif

#include <glimpse_imgui_shell.h>

struct gm_imgui_shell
{
    struct gm_logger *log;

    // Only set if user called _preinit_log_filename
    char *log_filename;

    // Note if NULL then that implies the user gave a custom logger and
    // we shouldn't destroy their logger.
    FILE *log_fp;

    char *app_name;
    char *app_title;

    char *custom_assets_root;

    enum gm_imgui_winsys winsys;

    enum gm_imgui_renderer requested_renderer;
    enum gm_imgui_renderer renderer;

    bool initialized;
    bool imgui_initialized;
    bool gl_initialized;

#ifdef SUPPORTS_GLFM
    GLFMDisplay *glfm_display;
#endif

#ifdef SUPPORTS_GLFW
    GLFWwindow *glfw_window;
#endif

    int surface_width;
    int surface_height;

    void (*log_ready_callback)(struct gm_imgui_shell *shell,
                               struct gm_logger *log,
                               void *user_data);
    void *log_ready_callback_data;

    void (*surface_created_callback)(struct gm_imgui_shell *shell,
                                     int width,
                                     int height,
                                     void *user_data);
    void *surface_created_callback_data;

    void (*surface_resized_callback)(struct gm_imgui_shell *shell,
                                     int width,
                                     int height,
                                     void *user_data);
    void *surface_resized_callback_data;

    void (*surface_destroyed_callback)(struct gm_imgui_shell *shell,
                                       void *user_data);
    void *surface_destroyed_callback_data;

    void (*app_focus_callback)(struct gm_imgui_shell *shell,
                               bool focused,
                               void *user_data);
    void *app_focus_callback_data;
    void (*mainloop_callback)(struct gm_imgui_shell *shell,
                              uint64_t timestamp,
                              void *user_data);
    void *mainloop_callback_data;
    void (*render_callback)(struct gm_imgui_shell *shell,
                              uint64_t timestamp,
                              void *user_data);
    void *render_callback_data;
};


