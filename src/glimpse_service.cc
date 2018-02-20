#include <jni.h>

static JavaVM *android_jvm_singleton;

extern "C" jint
JNI_OnLoad(JavaVM *vm, void *reserved)
{
    android_jvm_singleton = vm;

    return JNI_VERSION_1_6;
}
