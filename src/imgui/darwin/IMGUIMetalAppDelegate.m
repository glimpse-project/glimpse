
#import "IMGUIMetalAppDelegate.h"
#import "IMGUIMetalViewController.h"

#include "glimpse_imgui_darwin_shell.h"
#include "glimpse_imgui_shell_impl.h"

@implementation IMGUIMetalAppDelegate

// Ref: https://stackoverflow.com/questions/41317713/apple-metal-without-interface-builder

#if TARGET_OS_OSX
-(id)init
{
    if(self = [super init]) {
        struct gm_imgui_shell *shell = global_imgui_darwin_shell;

        gm_info(shell->log, "IMGUIMetalAppDelegate: init");

        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect contentSize = NSMakeRect(500.0, 500.0, 1000.0, 1000.0);
        NSUInteger windowStyleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskResizable | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
        self.window = [[NSWindow alloc] initWithContentRect:contentSize styleMask:windowStyleMask backing:NSBackingStoreBuffered defer:YES];
        self.window.backgroundColor = [NSColor whiteColor];
        self.window.title = @"Glimpse";

        [self.window setIsVisible: true];

        [self.window makeKeyAndOrderFront: self];
        //_appWindowController = [[AppWindowController alloc] initWithWindow:appWindow];
        //[_appWindowController showWindow:self];

        //if (self.window.isVisible)
        //    gm_info(shell->log, "Window visible");
        //else
        //    gm_info(shell->log, "Window not visible");

        NSMenu* bar = [[NSMenu alloc] init];
        [NSApp setMainMenu:bar];
        NSMenuItem* appMenuItem =
            [bar addItemWithTitle:@"" action:NULL keyEquivalent:@""];
        NSMenu* appMenu = [[NSMenu alloc] init];
        [appMenuItem setSubmenu:appMenu];

        [appMenu addItemWithTitle:[NSString stringWithFormat:@"About %@", @"Glimpse Shell"]
                           action:@selector(orderFrontStandardAboutPanel:)
                    keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        NSMenu* servicesMenu = [[NSMenu alloc] init];
        [NSApp setServicesMenu:servicesMenu];
        [[appMenu addItemWithTitle:@"Services"
                            action:NULL
                     keyEquivalent:@""] setSubmenu:servicesMenu];

        // Setup Preference Menu Action/Target on MainMenu
        NSMenu *mm = [NSApp mainMenu];
        NSMenuItem *myBareMetalAppItem = [mm itemAtIndex:0];
        NSMenu *subMenu = [myBareMetalAppItem submenu];
        NSMenuItem *prefMenu = [subMenu itemWithTag:100];
        prefMenu.target = self;
        prefMenu.action = @selector(showPreferencesMenu:);

        // Create a view
        //view = [[MTKView alloc] initWithFrame:CGRectMake(0, 0, 700, 700)];

        IMGUIMetalViewController *vc = [[IMGUIMetalViewController alloc] init];
        [self.window setContentViewController: vc];
    }
    return self;
}
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
-(void)applicationDidFinishLaunching:(NSNotification *)notification
{
    NSLog(@"show window");
    [self.window makeKeyAndOrderFront:self];     // Show the window
} 
#endif

@end
