
#import "IMGUIMetalRenderer.h"
#import <Metal/Metal.h>

#include "imgui.h"
#include "imgui_impl_metal.h"

#include "glimpse_imgui_darwin_shell.h"
#include "glimpse_imgui_shell_impl.h"

#if TARGET_OS_OSX
#include "imgui_impl_osx.h"
#endif

#include "glimpse_os.h"

@interface IMGUIMetalRenderer () {
#if TARGET_OS_OSX
    NSSize _oldSize;
#else
    CGSize _oldSize;
#endif
}
@property (nonatomic, strong) id <MTLDevice> device;
@property (nonatomic, strong) id <MTLCommandQueue> commandQueue;
@end

@implementation IMGUIMetalRenderer

-(nonnull instancetype)initWithView:(nonnull MTKView *)view;
{
    self = [super init];

    struct gm_imgui_shell *shell = global_imgui_darwin_shell;
    gm_info(shell->log, "IMGUIMetalRenderer: initWithView");

    if (self)
    {
        _oldSize = view.bounds.size;
        _device = view.device;
        _commandQueue = [_device newCommandQueue];

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplMetal_Init(_device);
    }

    return self;
}

- (void)drawInMTKView:(MTKView *)view
{
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize.x = view.bounds.size.width;
    io.DisplaySize.y = view.bounds.size.height;

    struct gm_imgui_shell *shell = global_imgui_darwin_shell;

#if TARGET_OS_OSX
    CGFloat framebufferScale = view.window.screen.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor;
#else
    CGFloat framebufferScale = view.window.screen.scale ?: UIScreen.mainScreen.scale;
#endif
    io.DisplayFramebufferScale = ImVec2(framebufferScale, framebufferScale);

#if 0
    gm_info(shell->log, "IMGUIMetalRenderer: drawInMTKView: scale=%f, x=%f,y-%f, width=%f,height=%f",
            framebufferScale,
            view.bounds.origin.x,
            view.bounds.origin.y,
            view.bounds.size.width,
            view.bounds.size.height);
#endif
    if (view.bounds.size.width != _oldSize.width ||
        view.bounds.size.height != _oldSize.height)
    {
        if (shell->surface_resized_callback) {
            shell->surface_resized_callback(shell,
                                            view.bounds.size.width,
                                            view.bounds.size.height,
                                            shell->surface_resized_callback_data);
        }
        _oldSize = view.bounds.size;
    }

    io.DeltaTime = 1 / float(view.preferredFramesPerSecond ?: 60);
    uint64_t time = gm_os_get_time();

    if (shell->mainloop_callback)
    {
        //ProfileScopedSection(MainAppLogic);

        shell->mainloop_callback(shell,
                                 time,
                                 shell->mainloop_callback_data);
    }

    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];

    static bool show_demo_window = false;
    static bool show_another_window = false;
    static float clear_color[4] = { 0.28f, 0.36f, 0.5f, 1.0f };

    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil)
    {
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

        // Here, you could do additional rendering work, including other passes as necessary.

        id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        [renderEncoder pushDebugGroup:@"ImGui demo"];

        // Start the Dear ImGui frame
        ImGui_ImplMetal_NewFrame(renderPassDescriptor);
#if TARGET_OS_OSX
        ImGui_ImplOSX_NewFrame(view);
#endif
        ImGui::NewFrame();


        if (shell->render_callback)
        {
            //ProfileScopedSection(AppRenderLogic);

            shell->render_callback(shell,
                                   time,
                                   shell->render_callback_data);
        }


        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        if (0)
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f    
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        ImDrawData *drawData = ImGui::GetDrawData();
        ImGui_ImplMetal_RenderDrawData(drawData, commandBuffer, renderEncoder);
        
        [renderEncoder popDebugGroup];
        [renderEncoder endEncoding];

        [commandBuffer presentDrawable:view.currentDrawable];
    }
    
    [commandBuffer commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size
{
}

@end
