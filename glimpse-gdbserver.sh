#!/bin/sh

# NB: First use the Android Developer option to set the app to wait for a
# debugger to attach on start up

PROCESS=com.impossible.glimpse.demo
ABI=`adb shell getprop ro.product.cpu.abi|dos2unix|head -1`
echo "ABI=\"$ABI\""

case "$ABI" in
    "arm64-v8a")
        echo "ARCH=android-arm64"
        ARCH=android-arm64
        ;;
    *)

        echo "Unhandled device architecture mapping (ABI=$ABI)"
        exit 1
esac

device_pid=`adb shell ps|grep $PROCESS|tail -1|awk '{ print $2; }'`
echo PID=$device_pid

# For gdbserver
adb -d forward tcp:5039 tcp:5039

adb push $ANDROID_NDK_HOME/prebuilt/$ARCH/gdbserver/gdbserver /data/local/tmp/gdbserver

# selinux will block us from running the gdbserver while under /data/local/
adb shell "run-as $PROCESS cp /data/local/tmp/gdbserver /data/data/$PROCESS/lib/gdbserver &"

adb shell run-as $PROCESS /data/data/$PROCESS/lib/gdbserver :5039 --attach $device_pid




