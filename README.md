Early playground for Glimpse Project

# Status

The early direction to get started with this project is to look at technically
implementing some of the capabilities demoed in this teaser video:

https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6

The first aspect looked at was supporting real-time (frontal) face detection
which we now have working - taking about 50 milliseconds on a Lenovo Phab 2.
From this a number of opportunities to accelerate detection on the GPU were
identified but for now these are being left for follow up later.

The current focus is supporting face features detection, locating the eyes and
mouth.  In itself this will allow us to visually augment faces, but note that
it doesn't give us a 3D pose estimation (though the feature locations should
help us estimate a pose later).

# Building Dependencies

So far this has two third_party build dependencies on DLib and OpenCV.

We aren't including the binaries in the repo for now otherwise the repo size
will quickly skyrocket.

(Currently assuming building on Linux...)


## DLib
Assuming $GLIMPSE_ROOT points to the top of the glimpse-sdk repo checkout...

```
git clone https://gitlab.com/impossible/dlib --branch=glimpse
cd dlib
mkdir build-release
cd build-release
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/dlib -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake ../
cmake --build .
cmake --build . --target install
```
*Note: DLib's build doesn't handle Android specifically and notably doesn't
place libraries into a per-abi directory so once we start building for multiple
ABIs we'll need to change this*

For face feature detection we're currently using this shape predictor data
which needs to be placed under the assets/ directory:
```
cd $GLIMPSE_ROOT/glimpse_android_demo/app/src/main/assets
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip shape_predictor_68_face_landmarks.dat.bz2
```

## OpenCV
```
git clone https://github.com/opencv/opencv
cd opencv
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/opencv -DWITH_TBB=ON -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake $@ ../
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


This repo has a scalar implementation of the same feature extraction algorithm used
in DLib that's a bit simpler to review for understanding what it's doing:
https://github.com/rbgirshick/voc-dpm (see features/features.cc) (though also
note there's the scalar code in fhog.h that has to handle the border pixels that
don't neatly fit into simd registers)

## Papers

[Histograms of Oriented Gradients for Human Detection by Navneet Dalal and Bill Triggs, CVPR 2005](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf)

[Object Detection with Discriminatively Trained Part Based Models by P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010](https://cs.brown.edu/~pff/papers/lsvm-pami.pdf)
