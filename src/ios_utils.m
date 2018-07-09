

@import UIKit;
@import AVFoundation;
@import CoreMotion;

#include <string.h>

#include <glimpse_log.h>
#include <glimpse_properties.h>
#include <glimpse_context.h>

#include "xalloc.h"
#include "ios_utils.h"

char *
ios_util_get_documents_path(void)
{
    NSString *documentsPath =
        [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                             NSUserDomainMask, YES) firstObject];

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

void
ios_log(const char *msg)
{
    NSLog(@"%@", @(msg));
}

enum gm_rotation
ios_get_device_rotation(void)
{
    switch ([[UIDevice currentDevice] orientation]) {
    case UIDeviceOrientationUnknown:
    case UIDeviceOrientationPortrait:
        return GM_ROTATION_0;
    case UIDeviceOrientationLandscapeRight:
        return GM_ROTATION_90;
    case UIDeviceOrientationPortraitUpsideDown:
        return GM_ROTATION_180;
    case UIDeviceOrientationLandscapeLeft:
        return GM_ROTATION_270;
    default:
        return GM_ROTATION_0;
    }
}

void
ios_begin_generating_device_orientation_notifications(void)
{
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
}

void
ios_end_generating_device_orientation_notifications(void)
{
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
}

@interface IOSAVSession : NSObject <AVCaptureDepthDataOutputDelegate,
                                    AVCaptureVideoDataOutputSampleBufferDelegate
                                    //AVCaptureDataOutputSynchronizerDelegate
                                    >
{
    struct gm_logger *log;

    AVCaptureSession *session;

    AVCaptureDeviceDiscoverySession *video_device_discovery_session;
    dispatch_queue_t session_queue;
    dispatch_queue_t data_queue;

    AVCaptureDevice *dual_cam_device;
    AVCaptureDeviceInput *dual_cam_device_input;
    AVCaptureDepthDataOutput *depth_output;
    AVCaptureVideoDataOutput *video_output;

    CMMotionManager *motion_manager;

    void (*configured_cb)(struct ios_av_session *session, void *user_data);
    void (*depth_cb)(struct ios_av_session *session,
                     struct gm_intrinsics *intrinsics,
                     float *acceleration,
                     int stride,
                     float *disparity,
                     void *user_data);
    void (*video_cb)(struct ios_av_session *session,
                     struct gm_intrinsics *intrinsics,
                     int stride,
                     uint8_t *video,
                     void *user_data);
    void *user_data;
}

- (void)depthDataOutput:(AVCaptureDepthDataOutput *)output
     didOutputDepthData:(AVDepthData *)depthData
              timestamp:(CMTime)timestamp
             connection:(AVCaptureConnection *)connection;
- (void)depthDataOutput:(AVCaptureDepthDataOutput *)output
       didDropDepthData:(AVDepthData *)depthData
              timestamp:(CMTime)timestamp
             connection:(AVCaptureConnection *)connection
                 reason:(AVCaptureOutputDataDroppedReason)reason;

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;
- (void)captureOutput:(AVCaptureOutput *)output
  didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;

//@property (nonatomic) dispatch_queue_t session_ueue;
//@property (nonatomic) AVCaptureSession *session;
@end

@implementation IOSAVSession

- (void)depthDataOutput:(AVCaptureDepthDataOutput *)output
     didOutputDepthData:(AVDepthData *)depthData
              timestamp:(CMTime)timestamp
             connection:(AVCaptureConnection *)connection
{
    if (depthData.depthDataType != kCVPixelFormatType_DisparityFloat32) {
        depthData = [depthData depthDataByConvertingToDepthDataType: kCVPixelFormatType_DisparityFloat32];
    }

    // TODO: compare with mapping r/w
    CVPixelBufferLockBaseAddress(depthData.depthDataMap,
                                 kCVPixelBufferLock_ReadOnly);
    void *base = CVPixelBufferGetBaseAddress(depthData.depthDataMap);
    int stride = CVPixelBufferGetBytesPerRow(depthData.depthDataMap);
    int w = CVPixelBufferGetWidth(depthData.depthDataMap);
    int h = CVPixelBufferGetHeight(depthData.depthDataMap);
    gm_debug(log, "depthDataOutput callback, depth = %p, w=%d, stride = %d, h=%d",
             base, w, stride, h);

    AVCameraCalibrationData *calib = depthData.cameraCalibrationData;
    gm_debug(log, "intrinsicMatrixReferenceDimensions w=%f, h=%f",
             calib.intrinsicMatrixReferenceDimensions.width,
             calib.intrinsicMatrixReferenceDimensions.height);
    float aspect_a = (float)w / (float)h;
    float aspect_b = (calib.intrinsicMatrixReferenceDimensions.width /
                      calib.intrinsicMatrixReferenceDimensions.height);
    gm_assert(log,
              aspect_a > (aspect_b - 5e-3) &&
              aspect_a < (aspect_b + 5e-3),
              "Intrinsics reference dimensions not consistent with buffer size");

    float scale_x = (float)w / calib.intrinsicMatrixReferenceDimensions.width;
    float scale_y = (float)h / calib.intrinsicMatrixReferenceDimensions.height;
    struct gm_intrinsics intrinsics;
    memset(&intrinsics, 0, sizeof(intrinsics));
    intrinsics.width = w;
    intrinsics.height = h;
    intrinsics.fx = calib.intrinsicMatrix.columns[0][0] * scale_x;
    intrinsics.fy = calib.intrinsicMatrix.columns[1][1] * scale_y;
    intrinsics.cx = calib.intrinsicMatrix.columns[2][0] * scale_x;
    intrinsics.cy = calib.intrinsicMatrix.columns[2][1] * scale_y;

    // Calculate the rotation to align with the ground
    CMAccelerometerData *data = self->motion_manager.accelerometerData;
    float acceleration[3];
    acceleration[0] = data.acceleration.x;
    acceleration[1] = data.acceleration.y;
    acceleration[2] = data.acceleration.z;

    self->depth_cb((__bridge struct ios_av_session *)self,
                   &intrinsics,
                   acceleration,
                   stride,
                   (float *)base,
                   self->user_data);
    CVPixelBufferUnlockBaseAddress(depthData.depthDataMap,
                                   kCVPixelBufferLock_ReadOnly);
}

- (void)depthDataOutput:(AVCaptureDepthDataOutput *)output
       didDropDepthData:(AVDepthData *)depthData
              timestamp:(CMTime)timestamp
             connection:(AVCaptureConnection *)connection
                 reason:(AVCaptureOutputDataDroppedReason)reason
{
    gm_debug(log, "depthDataOutput (dropped) callback");
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;
{
    gm_debug(log, "videoDataOutput callback");

    CVImageBufferRef ib = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (ib) {

        // TODO: compare with mapping r/w
        CVPixelBufferLockBaseAddress(ib,
                                     kCVPixelBufferLock_ReadOnly);
        void *base = CVPixelBufferGetBaseAddress(ib);
        int stride = CVPixelBufferGetBytesPerRow(ib);
        int w = CVPixelBufferGetWidth(ib);
        int h = CVPixelBufferGetHeight(ib);
        gm_debug(log, "video captureOutput callback, video = %p, w=%d, stride = %d, h=%d",
                 base, w, stride, h);

        struct gm_intrinsics intrinsics;
        memset(&intrinsics, 0, sizeof(intrinsics));
        intrinsics.width = w;
        intrinsics.height = h;

        CFTypeRef intrinsics_data_ref = CMGetAttachment(sampleBuffer,
                                                        kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                                                        nil);
        NSData *intrinsics_data = (__bridge NSData *)intrinsics_data_ref;
        gm_assert(log, intrinsics_data != NULL,
                  "Missing video sample buffer intrinsics");
        matrix_float3x3 *intrinsics_3x3 = (matrix_float3x3 *)intrinsics_data.bytes;

        intrinsics.fx = intrinsics_3x3->columns[0][0];
        intrinsics.fy = intrinsics_3x3->columns[1][1];
        intrinsics.cx = intrinsics_3x3->columns[2][0];
        intrinsics.cy = intrinsics_3x3->columns[2][1];

        self->video_cb((__bridge struct ios_av_session *)self,
                       &intrinsics,
                       stride,
                       (uint8_t *)base,
                       self->user_data);
        CVPixelBufferUnlockBaseAddress(ib,
                                       kCVPixelBufferLock_ReadOnly);
    } else {
        gm_debug(log, "no image buffer in video sampleBuffer");
    }
}

- (void)captureOutput:(AVCaptureOutput *)output
  didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;
{
    gm_debug(log, "videoDataOutput (dropped) callback");
}

- (void)configureSession
{
    NSError *error = nil;

    [self->session beginConfiguration];

    self->session.sessionPreset = AVCaptureSessionPreset640x480;
    //self->session.sessionPreset = AVCaptureSessionPreset1280x720; XXX: not getting depth with this preset

    self->dual_cam_device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInTrueDepthCamera
                                                               mediaType:AVMediaTypeVideo
                                                                position:AVCaptureDevicePositionFront];
    //self->dual_cam_device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInDualCamera
    //                                                           mediaType:AVMediaTypeVideo
    //                                                            position:AVCaptureDevicePositionBack];
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
        gm_debug(self->log, "Could not create video device input: %s",
                 [[error localizedDescription] UTF8String]);
        //self->setupResult = AVCamSetupResultSessionConfigurationFailed;
        [self->session commitConfiguration];
        return;
    }

    if ([self->session canAddInput:device_input]) {
        [self->session addInput:device_input];
        self->dual_cam_device_input = device_input;
    } else {
        gm_debug(self->log, "Could not add video device input to the session");
        //self->setupResult = AVCamSetupResultSessionConfigurationFailed;
        [self->session commitConfiguration];
        return;
    }

    AVCaptureVideoDataOutput *voutput = [[AVCaptureVideoDataOutput alloc] init];
    if ([self->session canAddOutput:voutput]) {

        [self->session addOutput:voutput];
        self->video_output = voutput;

        [voutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey]];

        gm_debug(self->log, "Added video output to the session");

        [voutput setSampleBufferDelegate: self queue:self->data_queue];

        AVCaptureConnection *conn = [voutput connectionWithMediaType:AVMediaTypeVideo];
        if (conn.cameraIntrinsicMatrixDeliverySupported)
            conn.cameraIntrinsicMatrixDeliveryEnabled = true;
        else {
            gm_debug(self->log, "Video camera doesn't support reporting intrinsics");
        }
#if 0 // deprecated api
        if (conn.isVideoMinFrameDurationSupported)
            conn.videoMinFrameDuration = CMTimeMake(1, 24);
        if (conn.isVideoMaxFrameDurationSupported)
            conn.videoMaxFrameDuration = CMTimeMake(1, 24);
#endif
    } else {
        gm_debug(self->log, "Could not add video output to the session");
    }

    AVCaptureDepthDataOutput *doutput = [[AVCaptureDepthDataOutput alloc] init];
    doutput.filteringEnabled = false;
    if ([self->session canAddOutput:doutput]) {

        [self->session addOutput:doutput];
        self->depth_output = doutput;

        gm_debug(self->log, "Added depth output to the session");

        [doutput setDelegate: self callbackQueue:self->data_queue];

        AVCaptureConnection *conn = [doutput connectionWithMediaType:AVMediaTypeDepthData];
        if (conn.cameraIntrinsicMatrixDeliverySupported)
            conn.cameraIntrinsicMatrixDeliveryEnabled = true;
        else {
            gm_debug(self->log, "Depth camera doesn't support reporting intrinsics");
        }
        conn.enabled = true;

#if 0 // deprecated api
        if (conn.isVideoMinFrameDurationSupported)
            conn.videoMinFrameDuration = CMTimeMake(1, 24);
        if (conn.isVideoMaxFrameDurationSupported)
            conn.videoMaxFrameDuration = CMTimeMake(1, 24);
#endif
    } else {
        gm_debug(self->log, "Could not add depth output to the session");
    }

    [self->session commitConfiguration];

    self->motion_manager = [[CMMotionManager alloc] init];

    if ([self->motion_manager isAccelerometerAvailable]) {
        self->motion_manager.accelerometerUpdateInterval = 0.01; // 100Hz updates
    } else {
        gm_debug(self->log, "Accelerometer unavailable");
    }
}

struct ios_av_session *
ios_util_av_session_new(struct gm_logger *log,
                        void (*configured_cb)(struct ios_av_session *session, void *user_data),
                        void (*depth_cb)(struct ios_av_session *session,
                                         struct gm_intrinsics *intrinsics,
                                         float *acceleration,
                                         int stride,
                                         float *disparity,
                                         void *user_data),
                        void (*video_cb)(struct ios_av_session *session,
                                         struct gm_intrinsics *intrinsics,
                                         int stride,
                                         uint8_t *video,
                                         void *user_data),
                        void *user_data)
{
    IOSAVSession *session = [[IOSAVSession alloc] init];

    session->log = log;
    session->configured_cb = configured_cb;
    session->depth_cb = depth_cb;
    session->video_cb = video_cb;
    session->user_data = user_data;

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

    session->data_queue = dispatch_queue_create("com.impossible.glimpse.av.data.queue",
                                                DISPATCH_QUEUE_SERIAL);
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo])
    {
    case AVAuthorizationStatusAuthorized:
        {
            gm_debug(log, "camera permissions authorized");
            break;
        }
    case AVAuthorizationStatusNotDetermined:
        {
            gm_debug(log, "camera permissions not determined");

            dispatch_suspend(session->session_queue);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^( BOOL granted ) {
                if ( ! granted ) {
                    //session->setupResult = AVCamSetupResultCameraNotAuthorized;
                }
                dispatch_resume(session->session_queue);
            }];
            break;
        }
    default:
        {
            gm_debug(log, "camera permissions denied");
            //session->setupResult = AVCamSetupResultCameraNotAuthorized;
            break;
        }
    }

    return (struct ios_av_session *)CFBridgingRetain(session);
}

void
ios_util_session_configure(struct ios_av_session *_session)
{
    IOSAVSession *session = (__bridge IOSAVSession *)_session;

    dispatch_async(session->session_queue, ^{
                       // FIXME - check status of permissions
                       gm_debug(session->log, "configureSession");
                       [session configureSession];
                       session->configured_cb(_session, session->user_data);
                   });
}

void
ios_util_session_start(struct ios_av_session *_session)
{
    IOSAVSession *session = (__bridge IOSAVSession *)_session;

    dispatch_async(session->session_queue, ^{
                       // FIXME - check status of configureSession
                       gm_debug(session->log, "startRunning");
                       [session->session startRunning];
                   });

    [session->motion_manager startAccelerometerUpdates];
}

void
ios_util_session_stop(struct ios_av_session *_session)
{
    IOSAVSession *session = (__bridge IOSAVSession *)_session;

    dispatch_async(session->session_queue, ^{
                       // FIXME - check status of configureSession
                       gm_debug(session->log, "stopRunning");
                       [session->session stopRunning];
                   });

    [session->motion_manager stopAccelerometerUpdates];
}

void
ios_util_session_destroy(struct ios_av_session *session)
{
    CFBridgingRelease(session);
}
@end


