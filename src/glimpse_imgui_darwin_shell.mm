/*
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

#import <TargetConditionals.h>

#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#else
#import <Cocoa/Cocoa.h>
#endif

#ifdef SUPPORTS_METAL
#include <IMGUIMetalAppDelegate.h>
#endif

#ifdef SUPPORTS_GLFM
#include <glfm_platform_ios.h>
#endif

#include "glimpse_imgui_shell_impl.h"
#include "glimpse_imgui_darwin_shell.h"

struct gm_imgui_shell *global_imgui_darwin_shell;

void
glimpse_imgui_darwin_metal_kit_main(struct gm_imgui_shell *shell)
{
#ifdef SUPPORTS_METAL
    gm_assert(shell->log,
              (shell->renderer == GM_IMGUI_RENDERER_METAL &&
               shell->winsys == GM_IMGUI_WINSYS_METAL_KIT),
              "Inconsistent renderer/winsys in %s", __func__);

    global_imgui_darwin_shell = shell;

    @autoreleasepool {
#if TARGET_OS_IOS
        UIApplicationMain(0, NULL, nil, NSStringFromClass([IMGUIMetalAppDelegate class]));
#else
        [NSApplication sharedApplication];

        IMGUIMetalAppDelegate *delegate = [[IMGUIMetalAppDelegate alloc] init];
        [NSApp setDelegate: delegate];

        gm_info(shell->log, "Running NSApp mainloop");
        [NSApp run];
#endif
    }
#endif // SUPPORTS_METAL
}

void
glimpse_imgui_darwin_glfm_main(struct gm_imgui_shell *shell)
{
#ifdef SUPPORTS_GLFM
    gm_assert(shell->log,
              (shell->renderer == GM_IMGUI_RENDERER_OPENGL &&
               shell->winsys == GM_IMGUI_WINSYS_GLFM),
              "Inconsistent renderer/winsys in %s", __func__);

    global_imgui_darwin_shell = shell;

#if TARGET_OS_IOS
    @autoreleasepool {
        UIApplicationMain(0, NULL, nil, NSStringFromClass([GLFMAppDelegate class]));
    }
#else
#error "GLFM unsupported on OSX"
#endif
#endif // SUPPORTS_GLFM
}
