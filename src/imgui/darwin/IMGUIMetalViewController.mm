
#import "IMGUIMetalViewController.h"
#import "IMGUIMetalRenderer.h"
#include "imgui.h"

#include "glimpse_imgui_darwin_shell.h"
#include "glimpse_imgui_shell_impl.h"

#if TARGET_OS_OSX
#include "imgui_impl_osx.h"
#endif

@interface IMGUIMetalViewController ()
@property (nonatomic, weak) MTKView *mtkView;
@property (nonatomic, strong) IMGUIMetalRenderer *renderer;
@end

@implementation IMGUIMetalViewController

- (MTKView *)mtkView {
    return _mtkView;
}

- (void)loadView
{
    struct gm_imgui_shell *shell = global_imgui_darwin_shell;

    gm_info(shell->log, "IMGUIMetalViewController: loadView");

    //MTKView* metalView = [[MTKView alloc] device:MTLCreateSystemDefaultDevice()];
    //MTKView* metalView = [[MTKView alloc] init];
    MTKView* metalView = [[MTKView alloc] initWithFrame:CGRectMake(0, 0, 700, 700)];

    [metalView setClearColor:MTLClearColorMake(0, 0, 0, 1)];
    [metalView setColorPixelFormat:MTLPixelFormatBGRA8Unorm];
    [metalView setDepthStencilPixelFormat:MTLPixelFormatDepth32Float];
    //[metalView setDelegate:self];

    self.mtkView = metalView;
    self.view = metalView;
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    struct gm_imgui_shell *shell = global_imgui_darwin_shell;

    gm_info(shell->log, "IMGUIMetalViewController: viewDidLoad");

    self.mtkView.device = MTLCreateSystemDefaultDevice();
    
    if (!self.mtkView.device) {
        NSLog(@"Metal is not supported");
        abort();
    }

    self.renderer = [[IMGUIMetalRenderer alloc] initWithView:self.mtkView];

    [self.renderer mtkView:self.mtkView drawableSizeWillChange:self.mtkView.bounds.size];

    self.mtkView.delegate = self.renderer;

#if TARGET_OS_OSX
    // Add a tracking area in order to receive mouse events whenever the mouse is within the bounds of our view
    NSTrackingArea *trackingArea = [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                                                options:NSTrackingMouseMoved | NSTrackingInVisibleRect | NSTrackingActiveAlways
                                                                  owner:self
                                                               userInfo:nil];
    [self.view addTrackingArea:trackingArea];
    
    // If we want to receive key events, we either need to be in the responder chain of the key view,
    // or else we can install a local monitor. The consequence of this heavy-handed approach is that
    // we receive events for all controls, not just ImGui widgets. If we had native controls in our
    // window, we'd want to be much more careful than just ingesting the complete event stream, though
    // we do make an effort to be good citizens by passing along events when ImGui doesn't want to capture.
    NSEventMask eventMask = NSEventMaskKeyDown | NSEventMaskKeyUp | NSEventMaskFlagsChanged | NSEventTypeScrollWheel;
    [NSEvent addLocalMonitorForEventsMatchingMask:eventMask handler:^NSEvent * _Nullable(NSEvent *event) {
        BOOL wantsCapture = ImGui_ImplOSX_HandleEvent(event, self.view);
        if (event.type == NSEventTypeKeyDown && wantsCapture) {
            return nil;
        } else {
            return event;
        }
        
    }];
    
    ImGui_ImplOSX_Init();

    if (shell->surface_resized_callback) {
        shell->surface_resized_callback(shell,
                                        self.mtkView.bounds.size.width,
                                        self.mtkView.bounds.size.height,
                                        shell->surface_resized_callback_data);
    }
#endif
}

#if TARGET_OS_OSX

- (void)mouseMoved:(NSEvent *)event {
    ImGui_ImplOSX_HandleEvent(event, self.view);
}

- (void)mouseDown:(NSEvent *)event {
    ImGui_ImplOSX_HandleEvent(event, self.view);
}

- (void)mouseUp:(NSEvent *)event {
    ImGui_ImplOSX_HandleEvent(event, self.view);
}

- (void)mouseDragged:(NSEvent *)event {
    ImGui_ImplOSX_HandleEvent(event, self.view);
}

- (void)scrollWheel:(NSEvent *)event {
    ImGui_ImplOSX_HandleEvent(event, self.view);
}

#elif TARGET_OS_IOS

// This touch mapping is super cheesy/hacky. We treat any touch on the screen
// as if it were a depressed left mouse button, and we don't bother handling
// multitouch correctly at all. This causes the "cursor" to behave very erratically
// when there are multiple active touches. But for demo purposes, single-touch
// interaction actually works surprisingly well.
- (void)updateIOWithTouchEvent:(UIEvent *)event {
    UITouch *anyTouch = event.allTouches.anyObject;
    CGPoint touchLocation = [anyTouch locationInView:self.view];
    ImGuiIO &io = ImGui::GetIO();
    io.MousePos = ImVec2(touchLocation.x, touchLocation.y);
    
    BOOL hasActiveTouch = NO;
    for (UITouch *touch in event.allTouches) {
        if (touch.phase != UITouchPhaseEnded && touch.phase != UITouchPhaseCancelled) {
            hasActiveTouch = YES;
            break;
        }
    }
    io.MouseDown[0] = hasActiveTouch;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

#endif

@end

