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
*Note: DLib's build doesn't handle Android specifically and notably doesn't place libraries into a per-abi directory so once we start building for multiple ABIs we'll need to change this*

=== OpenCV ===
```
git clone https://github.com/opencv/opencv
cd opencv
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$PWD/opencv-binaries -DWITH_TBB=ON -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake $@ ../
make -j8 && make install
```
*Note: see platforms/android/build_sdk.py and platforms/scripts/cmake_android_arm.sh to see how to match builds of official sdks*
