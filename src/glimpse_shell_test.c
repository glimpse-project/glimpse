#include <glimpse_log.h>
#include <glimpse_assets.h>
#include <glimpse_imgui_shell.h>

struct shell_test_app
{
    struct gm_imgui_shell *shell;
    struct gm_logger *log;
};

static void
on_log_ready_cb(struct gm_imgui_shell *shell,
                struct gm_logger *log,
                void *user_data)
{
    struct shell_test_app *app = user_data;

    // Use the shell's log for application logging too
    app->log = log;

    gm_info(log, "Log Ready Callback");
}

static void
on_surface_created_resized_cb(struct gm_imgui_shell *shell,
                              int width,
                              int height,
                              void *user_data)
{
    struct shell_test_app *app = user_data;

    gm_debug(app->log, "Surface Create/Resize Callback");
}

static void
on_surface_destroyed_cb(struct gm_imgui_shell *shell,
                        void *user_data)
{
    struct shell_test_app *app = user_data;

    gm_debug(app->log, "Surface Destroyed Callback");
}

static void
on_mainloop_cb(struct gm_imgui_shell *shell,
               uint64_t timestamp,
               void *user_data)
{
    struct shell_test_app *app = user_data;

    gm_debug(app->log, "Mainloop Callback");
}

static void
on_render_cb(struct gm_imgui_shell *shell,
             uint64_t timestamp,
             void *user_data)
{
    struct shell_test_app *app = user_data;

    gm_debug(app->log, "Render Callback");
}

void
glimpse_imgui_shell_main(struct gm_imgui_shell *shell,
                         int argc,
                         char **argv)
{
    struct shell_test_app *app = xcalloc(1, sizeof(*app));


    gm_imgui_shell_preinit_log_ready_callback(shell,
                                              on_log_ready_cb,
                                              app);

    gm_imgui_shell_preinit_surface_created_callback(shell,
                                                    on_surface_created_resized_cb,
                                                    app);
    gm_imgui_shell_preinit_surface_resized_callback(shell,
                                                    on_surface_created_resized_cb,
                                                    app);
    gm_imgui_shell_preinit_surface_destroyed_callback(shell,
                                                      on_surface_destroyed_cb,
                                                      app);
    gm_imgui_shell_preinit_mainloop_callback(shell,
                                             on_mainloop_cb,
                                             app);
    gm_imgui_shell_preinit_render_callback(shell,
                                           on_render_cb,
                                           app);

    gm_imgui_shell_init(shell,
                        "glimpse_shell_test",
                        "Glimpse Shell Test",
                        NULL); // abort on error
}
