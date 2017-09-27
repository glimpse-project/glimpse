Early playground for Glimpse Project

# Status

The early direction to get started with this project is to look at technically
implementing some of the capabilities demoed in this teaser video:

https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6

The first aspect looked at was supporting real-time (frontal) face detection
which we now have working well enough to move on:

https://medium.com/impossible/building-make-believe-tech-glimpse-in-progress-ecb9bbc113a1

There are still lots of opportunities to improve what we do for face tracking
but it's good enough to work with for now so we can start looking at the more
tricky problem of skeletal tracking.

The current focus is on skeletal tracking. The current aim is to reproduce the
R&D done by Microsoft for skeleton tracking with their Kinect cameras, which
provide similar data to Google Tango phones. Their research was published as a paper titled: [Real-Time Human Pose Recognition in Parts from Single Depth Images](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf)

In terms of understanding that paper then:

* Some reminders about set theory notation: http://byjus.com/maths/set-theory-symbols/
    * Also a reminder on set-builder notation: http://www.mathsisfun.com/sets/set-builder-notation.html
* How to calculate Shannon Entropy: http://www.bearcave.com/misl/misl_tech/wavelets/compression/shannon.html
* A key referenced paper on randomized trees for keypoint recognition: https://pdfs.semanticscholar.org/f637/a3357444112d0d2c21765949409848a5cba3.pdf
    * A related technical report (more implementation details): http://cvlabwww.epfl.ch/~lepetit/papers/lepetit_tr04.pdf
* Referenced paper on using meanshift: https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf
    * Simple intuitive description of meanshift: http://docs.opencv.org/trunk/db/df8/tutorial_py_meanshift.html
    * Comparison of mean shift tracking methods (inc. camshift): http://old.cescg.org/CESCG-2008/papers/Hagenberg-Artner-Nicole.pdf


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
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/opencv -DWITH_TBB=ON -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake ../
make -j8 && make install
```
*Note: see platforms/android/build_sdk.py and
platforms/scripts/cmake_android_arm.sh to see how to match builds of official
sdks*


## libpng
*Note: zlib (which libpng depends on) is provided by the NDK itself)*

```
git clone https://github.com/glennrp/libpng -b libpng16
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/libpng -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -Dld-version-script=OFF -DPNG_ARM_NEON=on ../
make -j8
make install
```


## Boost

libpcl depends on a number of Boost libraries...

Note: I had to avoid using the latest Boost because the FindBoost.cmake script
that was shipped with the latest cmake packaged for Ubuntu didn't know about
Boost versions > 1.61

```
git clone --recursive https://github.com/boostorg/boost.git -b boost-1.61.0
cd boost
./bootstrap.sh
PATH=`/path/to/standalone/android-toolchain/bin`:$PATH ./b2 install -j8 -q \
    toolset=clang \
    link=static \
    target-os=android \
    --prefix=$GLIMPSE_ROOT/third_party/boost \
    --with-system \
    --with-filesystem \
    --with-date_time \
    --with-thread \
    --with-iostreams \
    --with-serialization \
    -sNO_BZIP2=1
```

## libflann

```
git clone git://github.com/mariusmuja/flann.git
cd flann
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/libflann -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF -DBUILD_TESTS=OFF -DUSE_OPENMP=OFF ../
make
make install
```

## Eigen

```
hg clone https://bitbucket.org/eigen/eigen/
cd eigen
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/eigen -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake ../
make
make install
```

## QHull

The upstream qhull project has broken a cmake build which is fixed by this branch...
```
git clone https://github.com/jamiesnape/qhull -b cmake
cd qhull
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/libqhull -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake ../
make
make install
```

## libpcl

Note: libpcl itself depends on Boost, libeigen, libflann, libqhull and libpng

If building on Ubuntu for development, you can build libpcl without visualization support like:
```
# apt-get install libflann-dev libqhull-dev libboost-all-dev
$ git clone https://github.com/PointCloudLibrary/pcl
$ cd pcl
$ mkdir build && cd build
$ cmake ../ -DWITH_OPENGL=OFF -DWITH_PCAP=OFF
$ make -j8
```

For an Android build (assuming the prerequisities have been built as above) do:

```
git clone https://github.com/PointCloudLibrary/pcl
cd pcl
mkdir build-release
cd build-release
cmake -DCMAKE_INSTALL_PREFIX=$GLIMPSE_ROOT/third_party/libpcl -DCMAKE_INSTALL_LIBDIR=lib/arm64-v8a -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_NDK=$ANDROID_NDK_HOME -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DCMAKE_FIND_ROOT_PATH="$GLIMPSE_ROOT/third_party/eigen;$GLIMPSE_ROOT/third_party/libflann;$GLIMPSE_ROOT/third_party/boost;$GLIMPSE_ROOT/third_party/libqhull;$GLIMPSE_ROOT/third_party/libpng" -DWITH_OPENGL=OFF -DWITH_PCAP=OFF -DWITH_OPENNI2=OFF ../
make -j8
make install
cd $GLIMPSE_ROOT/third_party/libpcl/lib
ln -s . arm64-v8a
```

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
