#pragma once

#include <stdint.h>
#include <stdbool.h>

#include <glimpse_log.h>

#ifdef __cplusplus
extern "C" {
#endif

enum gm_imgui_renderer
{
    GM_IMGUI_RENDERER_AUTO,
    GM_IMGUI_RENDERER_OPENGL,
    GM_IMGUI_RENDERER_METAL,
};

enum gm_imgui_winsys
{
    GM_IMGUI_WINSYS_NONE,
    GM_IMGUI_WINSYS_GLFM,
    GM_IMGUI_WINSYS_GLFW,
    GM_IMGUI_WINSYS_METAL_KIT,
};

struct gm_imgui_shell;

// If you want to create your own logger that the shell will use then call this
// api to attach your custom logger to the shell before it's fully
// initialized...  (Normally gm_imgui_shell_init will automatically create a
// gm_logger).
void
gm_imgui_shell_preinit_log(struct gm_imgui_shell *shell,
                           struct gm_logger *log);

// If you want to override what file the shell opens for logging, use
// this api to set the log's filename
void
gm_imgui_shell_preinit_log_filename(struct gm_imgui_shell *shell,
                                    const char *log_filename);

// As the shell is initialized it may itself need to read application assets
// (such as loading a default font) and the 'assets root' is normally
// determined automatically according to the platform we are running on. If
// you want to override where assets are loaded from then call this
// before the shell is fully initialized...
void
gm_imgui_shell_preinit_assets_root(struct gm_imgui_shell *shell,
                                   const char *assets_root);

// On platforms where multiple graphics apis are supported then this api can
// affect the choice of API. By default Metal is used if supported otherwise
// OpenGL[ES] is used.
//
// This will return false if the requested API is not supported.
bool
gm_imgui_shell_preinit_renderer(struct gm_imgui_shell *shell,
                                enum gm_imgui_renderer renderer);

void
gm_imgui_shell_preinit_log_ready_callback(struct gm_imgui_shell *shell,
                                          void (*callback)(struct gm_imgui_shell *shell,
                                                           struct gm_logger *log,
                                                           void *user_data),
                                          void *user_data);

void
gm_imgui_shell_preinit_surface_created_callback(struct gm_imgui_shell *shell,
                                                void (*callback)(struct gm_imgui_shell *shell,
                                                                 int width,
                                                                 int height,
                                                                 void *user_data),
                                                void *user_data);
void
gm_imgui_shell_preinit_surface_resized_callback(struct gm_imgui_shell *shell,
                                                void (*callback)(struct gm_imgui_shell *shell,
                                                                 int width,
                                                                 int height,
                                                                 void *user_data),
                                                void *user_data);
void
gm_imgui_shell_preinit_surface_destroyed_callback(struct gm_imgui_shell *shell,
                                                  void (*callback)(struct gm_imgui_shell *shell,
                                                                   void *user_data),
                                                  void *user_data);
void
gm_imgui_shell_preinit_app_focus_callback(struct gm_imgui_shell *shell,
                                          void (*callback)(struct gm_imgui_shell *shell,
                                                           bool focused,
                                                           void *user_data),
                                          void *user_data);
void
gm_imgui_shell_preinit_mainloop_callback(struct gm_imgui_shell *shell,
                                         void (*callback)(struct gm_imgui_shell *shell,
                                                          uint64_t timestamp,
                                                          void *user_data),
                                         void *user_data);
void
gm_imgui_shell_preinit_render_callback(struct gm_imgui_shell *shell,
                                       void (*callback)(struct gm_imgui_shell *shell,
                                                        uint64_t timestamp,
                                                        void *user_data),
                                       void *user_data);


// Early on without glimpse_imgui_shell_main() your application should
// complete the initialization of the given shell by calling this api.
// (optionally using any _preinit apis as necessary to override the
// default behaviour of initialization)
bool
gm_imgui_shell_init(struct gm_imgui_shell *shell,
                    const char *app_name,
                    const char *app_title,
                    char **err);

// Call this to be able to share the same logger as the shell
struct gm_logger *
gm_imgui_shell_get_log(struct gm_imgui_shell *shell);

// Call to know what renderer was finally chosen if a renderer wasn't
// explicitly requested.
enum gm_imgui_renderer
gm_imgui_shell_get_renderer(struct gm_imgui_shell *shell);

// If the renderer == _OPENGL then this can be called to clarify if
// it's big GL or GLES.
bool
gm_imgui_shell_is_gles_renderer(struct gm_imgui_shell *shell);

// E.g. on Android and iOS then applications are required to inset their
// content, either to account for physical notches in the display for
// for status areas.
void
gm_imgui_shell_get_chrome_insets(struct gm_imgui_shell *shell,
                                 float *top,
                                 float *right,
                                 float *bottom,
                                 float *left);

// Each application using this imgui shell needs to implement this
// function where it can parse command line arguments (if command
// line arguments aren't applicable for the current platform then
// argc will == 0 and argv may be NULL)
extern void
glimpse_imgui_shell_main(struct gm_imgui_shell *shell,
                         int argc,
                         char **argv);

#ifdef __cplusplus
}
#endif
