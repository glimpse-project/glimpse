#!/bin/sh


$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/gdb \
    ./phab2-sysroot/system/bin/app_process64 \
    -ex "set sysroot $PWD/phab2-sysroot" \
    -ex "set solib-search-path ./glimpse_android_demo/app/build/intermediates/cmake/debug/obj/arm64-v8a:./tango_client_api/lib/arm64-v8a:./tango_support_api/lib/arm64-v8a" \
    -ex "dir ./tango_gl/src" \
    -ex "dir ./glimpse_android_demo/app/src/main/jni" \
    -ex "target remote localhost:5039"
