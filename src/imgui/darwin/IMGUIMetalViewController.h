
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import "IMGUIMetalRenderer.h"

#if TARGET_OS_IPHONE

#import <UIKit/UIKit.h>

@interface IMGUIMetalViewController : UIViewController
@end

#else

#import <Cocoa/Cocoa.h>

@interface IMGUIMetalViewController : NSViewController
@end

#endif
