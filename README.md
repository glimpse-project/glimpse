Early playground for Glimpse Project


== Building Dependencies ==

So far this has two third_party build dependencies on DLib and OpenCV.

We aren't including the binaries in the repo for now otherwise the repo size
will quickly skyrocket.

(Currently assuming building on Linux...)

=== DLib ===
```
git clone https://github.com/davisking/dlib
cd dlib
mkdir build-release
cd build-release
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$PWD/dlib-binaries -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake ../
cmake --build .
cmake --build . --target install
```
*Note: DLib's build doesn't handle Android specifically and notably doesn't
place libraries into a per-abi directory so once we start building for multiple
ABIs we'll need to change this*

=== OpenCV ===
```
git clone https://github.com/opencv/opencv
cd opencv
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$PWD/opencv-binaries -DWITH_TBB=ON -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake $@ ../
make -j8 && make install
```
*Note: see platforms/android/build_sdk.py and
platforms/scripts/cmake_android_arm.sh to see how to match builds of official
sdks*


=== Misc Android development notes / reminders ===

To install a built debug apk explicitly either do `adb install -r
./app/build/outputs/apk/app-debug.apk` or `./gradlew installDebug`

To run the glimpse demo via adb:
`adb shell am start com.impossible.glimpse.demo/com.impossible.glimpse.demo.MainActivity`

Assuming the app of interest has been set selected under the devices
`Developer options -> Select debug app` menu and `Wait for debugger` is enabled
then it's necessary to attach JDB after invoking `am start` for the app to
continue...

Run `./glimpse-jdb.sh` to attach and continue executing.

Run `./glimose-jdb.sh --break` to attach JDB but break before any native shared
libraries are loaded so it's possible to also attach `gdb` before any of our
native code has had a chance to blow up.

Run gdbserver on the device, attached to our process, using the helper script
`./glimpse-gdbserver.sh` - this will forward tcp:5039 to localhost

Connect gdb to the device with `./glimpse-gdb.sh`
