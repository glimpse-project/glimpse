Early playground for Glimpse Project

# Status

The early direction to get started with this project is to look at technically
implementing some of the capabilities demoed in this teaser video:

https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6

For now the focus is on bootstrapping frontal face detection that can run at
a decent rate on a Lenovo Phab 2 phone based on the YUV camera frames we can
access via the Tango C SDK.

There are some well researched methods for handling face detection so we don't
need to invent a new wheel here but the tricky bit is getting something running
at a decent rate on a phone.

We're starting with DLib's HOG (Histogram Of Gradients) detection
implementation. HOG filters have some known limiations when it comes to
detection of rotated objects, but aren't too computationally expensive so it
should be possible run in real time on the Phab 2 device we're currently
targetting. It generally seems like HOG detection works better than the older
Haar/Viola-Jones method implemented by OpenCV.

Unfortunately even though DLib's HOG filter is considered to be
'fast'/'efficient' that's based on running with a fairly high-end PC hardware,
e.g. with AVX/SSE4 instructions. The implementation doesn't have ARM/NEON or GPU
optimizations. On a Phab 2 the out-of-the-box face detector is liable to take
around a minute on a 1920x1080 greyscale (just the luminance from a yuv buffer)
camera frame. Downscaling to 960x540 first it still takes ~20 seconds to run.

After adding some instrumentation to DLib this is how I found the time is spent:

Firstly before we hand anything to DLib it takes ~230ms to downsample (bilinear,
without using NEON) to 960x540 on the cpu (single threaded).

Internally Dlib will also do it's own downsampling to build a pyramid of
images during face detection (each level is 5/6ths the size of the previous)
Given a 960x540 image it will downsample 10 times with these timings:

```
 1)  1.202s
 2)  829.238ms
 3)  577.865ms
 4)  397.632ms
 5)  275.419ms
 6)  188.941ms
 7)  131.361ms
 8)  90.234ms
 9)  61.334ms
 10) 42.011ms
```

(taking ~8 seconds)

After downsampling it then takes about 2.5 seconds per HOG filter and Dlib's
out-of-the-box face detector is based on five separately trained filters: A
front looking, left looking, right looking, front looking but rotated left, and
finally a front looking but rotated right one.

(taking ~12.5 seconds)

So TL;DR we need to look into what opportunities we have to optimize this :)


## Current Optimization Plan

1. The first low-hanging optimization is to drop all but the front facing HOG
filter and accept the limitation. We can potentially be more adaptive later,
enabling more filters if we have smaller regions to focus on. This cuts the
time in half to ~10 seconds.

2. Use the GPU to do all the bilinear downsampling. No numbers yet for the
benefit but depending on how lucky we can get with avoiding copies getting
the data in and out of the GPU (not 100% sure how awkward that'll be on
Android) then I'd expect it'll be more effective than going for a NEON optimized
path. Of course this contends for the GPU resources but I don't currently see
us getting too fancy on the rendering side of things.

3. Understand if the next steps of evaluating the filter across all levels
of the pyramid can also be done on the GPU. I (possibly naively) assume it
boils down to a heirachy of convoltion filters that could be computed reasonly
efficiently on the GPU.

4. Experiment with fewer pyramid levels, by downsampling at a faster rate.

5. Experiment with fewer pyramid levels, by skipping the last few levels if we
maybe don't expect faces to fill close to 100% of the frame. (I guess there's
not much benefit skipping the smaller levels though)

6. Experiment with training a smaller HOG filter (say 60x60) instead of the
pre-trained 80x80 filter which would in turn let us downsample the first level
of the pyramid even further.

7. Experiment with masking where we scan in the larger levels, making some
assumptions that we aren't looking for small faces near the bottom of the
screen if our use case is also expecting to track someones body which would
implicitly be offscreen.

8. Make sure we scan the smallest pyramid levels first so that faces detected
at these levels can mask out large regions to ignore in the larger levels.


# Building Dependencies

So far this has two third_party build dependencies on DLib and OpenCV.

We aren't including the binaries in the repo for now otherwise the repo size
will quickly skyrocket.

(Currently assuming building on Linux...)


## DLib
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

*FIXME: actually we're using a modified branch of Dlib, not the upstream master
branch*


## OpenCV
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


# Misc Android development notes / reminders

To install a built debug apk explicitly either do `adb install -r
./app/build/outputs/apk/app-debug.apk` or `./gradlew installDebug` (the latter
will also re-build as appropriate)

To run the glimpse demo via adb:
`adb shell am start com.impossible.glimpse.demo/com.impossible.glimpse.demo.MainActivity`

Assuming the app of interest has been chosen under the device's
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


# References

https://github.com/betars/Face-Resources

## Papers

[Histograms of Oriented Gradients for Human Detection by Navneet Dalal and Bill Triggs, CVPR 2005](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf)

[Object Detection with Discriminatively Trained Part Based Models by P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010](https://cs.brown.edu/~pff/papers/lsvm-pami.pdf)
