
#import <MetalKit/MetalKit.h>

@interface IMGUIMetalRenderer : NSObject <MTKViewDelegate>

-(nonnull instancetype)initWithView:(nonnull MTKView *)view;

@end

