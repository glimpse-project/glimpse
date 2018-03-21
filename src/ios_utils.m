

#import <UIKit/UIKit.h>

char *
ios_util_get_documents_path(void)
{
    //NSFileManager *fileManager = [NSFileManager defaultManager];
#if 0
    NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];

    return strdup([documentsPath UTF8String]);
#endif
    return strdup("");
}
