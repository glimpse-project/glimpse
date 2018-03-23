

@import UIKit;
@import AVFoundation;

#include <string.h>

#include <glimpse_log.h>

#include "xalloc.h"

char *
ios_util_get_documents_path(void)
{
    NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];

    return strdup([documentsPath UTF8String]);
}

char *
ios_util_get_resources_path(void)
{
    char path[PATH_MAX];

    CFBundleRef bundle = CFBundleGetMainBundle();
    if (bundle) {
        CFURLRef resourcesURL = CFBundleCopyResourcesDirectoryURL(bundle);
        if (resourcesURL) {
            Boolean success = CFURLGetFileSystemRepresentation(resourcesURL, TRUE, (UInt8 *)path,
                                                               sizeof(path) - 1);
            CFRelease(resourcesURL);
            if (success) {
                return strdup(path);
            }
        }
    }

    return NULL;
}

@interface IOSAVSession : NSObject
{
    struct gm_logger *log;

    AVCaptureSession *session;

    AVCaptureDeviceDiscoverySession *video_device_discovery_session;
    dispatch_queue_t session_queue;

    AVCaptureDevice *dual_cam_device;
    AVCaptureDeviceInput *dual_cam_device_input;
}
//@property (nonatomic) dispatch_queue_t session_ueue;
//@property (nonatomic) AVCaptureSession *session;
@end

@implementation IOSAVSession

- (void)configureSession
{
    NSError *error = nil;

    [self->session beginConfiguration];

    self->dual_cam_device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInDualCamera
                                                                  mediaType:AVMediaTypeVideo
                                                                   position:AVCaptureDevicePositionBack];
    if (!self->dual_cam_device) {
        gm_debug(self->log, "Failed to find dual camera device");
        //self->setupResult = AVCamSetupResultSessionConfigurationFailed;
        [self->session commitConfiguration];
        return;
    } else {
        gm_debug(self->log, "Found dual camera device");
    }

    AVCaptureDeviceInput *device_input =
        [AVCaptureDeviceInput deviceInputWithDevice:self->dual_cam_device
                                              error:&error];
    if (!device_input) {
        gm_debug(self->log, "Could not create video device input: %s", [[error localizedDescription] UTF8String]);
        //self->setupResult = AVCamSetupResultSessionConfigurationFailed;
        [self->session commitConfiguration];
        return;
    }

    if ([self->session canAddInput:dual_cam_device_input]) {

        [self->session addInput:dual_cam_device_input];
        self->dual_cam_device_input = device_input;

        dispatch_async(dispatch_get_main_queue(), ^{
            /*
               Why are we dispatching this to the main queue?
               Because AVCaptureVideoPreviewLayer is the backing layer for AVCamPreviewView and UIView
               can only be manipulated on the main thread.
Note: As an exception to the above rule, it is not necessary to serialize video orientation changes
on the AVCaptureVideoPreviewLayer’s connection with other session manipulation.

Use the status bar orientation as the initial video orientation. Subsequent orientation changes are
handled by -[AVCamCameraViewController viewWillTransitionToSize:withTransitionCoordinator:].
*/
            UIInterfaceOrientation statusBarOrientation = [UIApplication sharedApplication].statusBarOrientation;
            AVCaptureVideoOrientation initialVideoOrientation = AVCaptureVideoOrientationPortrait;
            if ( statusBarOrientation != UIInterfaceOrientationUnknown ) {
                initialVideoOrientation = (AVCaptureVideoOrientation)statusBarOrientation;
            }

            //self.previewView.videoPreviewLayer.connection.videoOrientation = initialVideoOrientation;
        } );
    }
    else {
        gm_debug(self->log, "Could not add video device input to the session");
        //self->setupResult = AVCamSetupResultSessionConfigurationFailed;
        [self->session commitConfiguration];
        return;
    }

    [self->session commitConfiguration];
}

struct ios_av_session *
ios_util_av_session_new(struct gm_logger *log)
{
    IOSAVSession *session = [[IOSAVSession alloc] init];

    session->log = log;

    session->session = [[AVCaptureSession alloc] init];

    NSArray<AVCaptureDeviceType> *device_types =
        @[AVCaptureDeviceTypeBuiltInWideAngleCamera,
          AVCaptureDeviceTypeBuiltInDualCamera];

    session->video_device_discovery_session =
        [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:device_types
                                                               mediaType:AVMediaTypeVideo
                                                                position:AVCaptureDevicePositionUnspecified];

    session->session_queue = dispatch_queue_create("com.impossible.glimpse.av.session.queue",
                                                   DISPATCH_QUEUE_SERIAL);

    /*
       Check video authorization status. Video access is required and audio
       access is optional. If audio access is denied, audio is not recorded
       during movie recording.
       */
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo])
    {
    case AVAuthorizationStatusAuthorized:
        {
            // The user has previously granted access to the camera.
            gm_debug(log, "camera permissions authorized");
            break;
        }
    case AVAuthorizationStatusNotDetermined:
        {
            gm_debug(log, "camera permissions not determined");

            /*
               The user has not yet been presented with the option to grant
               video access. We suspend the session queue to delay session
               setup until the access request has completed.

               Note that audio access will be implicitly requested when we
               create an AVCaptureDeviceInput for audio during session setup.
               */
            dispatch_suspend(session->session_queue);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^( BOOL granted ) {
                if ( ! granted ) {
                    //session->setupResult = AVCamSetupResultCameraNotAuthorized;
                }
                dispatch_resume( session->session_queue );
            }];
            break;
        }
    default:
        {
            gm_debug(log, "camera permissions denied");

            // The user has previously denied access.
            //session->setupResult = AVCamSetupResultCameraNotAuthorized;
            break;
        }
    }

    /*
       Setup the capture session.
       In general it is not safe to mutate an AVCaptureSession or any of its
       inputs, outputs, or connections from multiple threads at the same time.

       Why not do all of this on the main queue?
       Because -[AVCaptureSession startRunning] is a blocking call which can
       take a long time. We dispatch session setup to the sessionQueue so
       that the main queue isn't blocked, which keeps the UI responsive.
       */
    dispatch_async( session->session_queue, ^{
                        [session configureSession];
                    });

    return (struct ios_av_session *)CFBridgingRetain(session);
}

@end
