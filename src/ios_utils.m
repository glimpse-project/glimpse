

#import <UIKit/UIKit.h>

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
