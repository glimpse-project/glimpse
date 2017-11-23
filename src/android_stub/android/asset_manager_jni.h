#pragma once

#include <jni.h>
#include <android/asset_manager.h>

static AAssetManager *
AAssetManager_fromJava (JNIEnv *env, jobject assetManager)
    __attribute__((unused));

static AAssetManager *
AAssetManager_fromJava (JNIEnv *env, jobject assetManager)
{
    return NULL;
}
