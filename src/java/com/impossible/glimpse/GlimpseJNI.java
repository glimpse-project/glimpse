/*
 * Copyright 2015 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.impossible.glimpse;

import android.app.Activity;
import android.content.res.AssetManager;
import android.os.IBinder;
import android.util.Log;

import java.io.File;

public class GlimpseJNI
{
    public static final int ARCH_ERROR = -2;
    public static final int ARCH_FALLBACK = -1;
    public static final int ARCH_DEFAULT = 0;
    public static final int ARCH_ARM64 = 1;
    public static final int ARCH_ARM32 = 2;
    public static final int ARCH_X86_64 = 3;
    public static final int ARCH_X86 = 4;

    public static final int loadTangoSharedLibrary() {
        int loadedSoId = ARCH_ERROR;
        String basePath = "/data/data/com.google.tango/libfiles/";
        if (!(new File(basePath).exists())) {
            basePath = "/data/data/com.projecttango.tango/libfiles/";
        }
        Log.i("GlimpseJNI", "loadTangoSharedLibrary: basePath: " + basePath);

        try {
            System.load(basePath + "arm64-v8a/libtango_client_api.so");
            loadedSoId = ARCH_ARM64;
            Log.i("GlimpseJNI", "Success! Using arm64-v8a/libtango_client_api.");
        } catch (UnsatisfiedLinkError e) {
        }
        if (loadedSoId < ARCH_DEFAULT) {
            try {
                System.load(basePath + "armeabi-v7a/libtango_client_api.so");
                loadedSoId = ARCH_ARM32;
                Log.i("GlimpseJNI", "Success! Using armeabi-v7a/libtango_client_api.");
            } catch (UnsatisfiedLinkError e) {
            }
        }
        if (loadedSoId < ARCH_DEFAULT) {
            try {
                System.load(basePath + "x86_64/libtango_client_api.so");
                loadedSoId = ARCH_X86_64;
                Log.i("GlimpseJNI", "Success! Using x86_64/libtango_client_api.");
            } catch (UnsatisfiedLinkError e) {
            }
        }
        if (loadedSoId < ARCH_DEFAULT) {
            try {
                System.load(basePath + "x86/libtango_client_api.so");
                loadedSoId = ARCH_X86;
                Log.i("GlimpseJNI", "Success! Using x86/libtango_client_api.");
            } catch (UnsatisfiedLinkError e) {
            }
        }
        if (loadedSoId < ARCH_DEFAULT) {
            try {
                System.load(basePath + "default/libtango_client_api.so");
                loadedSoId = ARCH_DEFAULT;
                Log.i("GlimpseJNI", "Success! Using default/libtango_client_api.");
            } catch (UnsatisfiedLinkError e) {
            }
        }
        if (loadedSoId < ARCH_DEFAULT) {
            try {
                System.loadLibrary("tango_client_api");
                loadedSoId = ARCH_FALLBACK;
                Log.i("GlimpseJNI", "Falling back to libtango_client_api.so symlink.");
            } catch (UnsatisfiedLinkError e) {
            }
        }
        return loadedSoId;
    }

    public static native void onTangoServiceConnected(IBinder nativeTangoServiceBinder);
}
