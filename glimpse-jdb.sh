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

# This is the static {} class initializer where we call loadLibrary to
# load our native shared libraries...
JAVA_BREAKPOINT="com.impossible.glimpse.demo.TangoJNINative.<clinit>"


device_pid=`adb shell ps|grep $PROCESS|tail -1|awk '{ print $2; }'`
echo PID=$device_pid

local_jdb_port=65534
adb -d forward tcp:$local_jdb_port jdwp:$device_pid

# This is *really* nasty but jdb doesn't have a command line option for initial
# commands to run once attached. Further the Android feature to wait for a
# debugger to attach will immediately resume the application ~1.5seconds after
# the debugger attaches so it's important to set breakpoints asap after
# attaching.
#
# Seriously the native development experience on Android is *soooo*
# horrible!!!
echo "stop in $JAVA_BREAKPOINT">~/.jdbrc

jdb -attach localhost:$local_jdb_port
#jdb -connect com.sun.jdi.SocketAttach:hostname=localhost,$local_jdb_port



