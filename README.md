[![Build Status](https://travis-ci.org/glimpse-project/glimpse.svg?branch=master)](https://travis-ci.org/glimpse-project/glimpse)

# Glimpse — Communicating with the richness of our physical selves

We are building a real-time motion capture system for the body using mobile
devices such as phones and AR/VR headsets.

It's important to us that this kind of technology not be locked within walled
gardens as the ability to interact and express with our bodies is most
interesting when that can include everyone.

Our rendering pipeline (for generating training data) is based on open source
projects including [Blender](https://www.blender.org) and
[Makehuman](https://www.makehuman.org), and all our code for training and
runtime motion capture is under the permissive
[MIT](https://en.wikipedia.org/wiki/MIT_License) license.


# Status

The early direction of this project has been to implement some of the
capabilities demoed in our
['Glimpse — a sneak peek into your creative self'](https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6)
teaser video.

It's early days but we already have an end-to-end MoCap system working,
including:

* A rendering pipeline implemented using Blender to create large sets of
  training data along with a post-processing tool that helps model the
  artefacts of real-world cameras.
* A custom framework for training a forest of random decision trees that's
  tailored and optimized for our needs such that we can efficiently train trees
  with more than 1M images on very modest PC hardware.
* A runtime engine useable on Linux, OSX, Android and iOS supporting:
  *  Multiple camera device backends (Kinect, Tango, AVFoundation)
  *  Real-time segmentation of people from the background (including support
     for a moving camera)
  *  Real-time inference of per-pixel, body part labels (currently 34)
  *  Real-time inference of a 3D skeleton based on body part labels
* Development tooling to capture recordings for offline testing and
  visualizing the tracking engine state including 3D point cloud views of camera
  data and wireframes of inferred skeletons.
* A native Unity plugin and editor extensions to facilitate easily creating
  content based on the skeltal data from Glimpse.

Here's a recording from March 2018 of Glimpse running on an Android, Zenfone AR:
[![](http://i.vimeocdn.com/video/686380376_640.jpg)](https://vimeo.com/258250667)

and a screenshot of the Glimpse Viewer debug tool:
![](https://raw.githubusercontent.com/wiki/glimpse-project/glimpse/images/screenshot-2018-07-05.png)


# Fetching

Along with this repository you'll also need to fetch the [Glimpse Training
Data](https://github.com/glimpse-project/glimpse-training-data) and/or the
[Glimpse Pre-Trained Models](https://github.com/glimpse-project/glimpse-models)
as follows:

```
git clone https://github.com/glimpse-project/glimpse
git clone --depth=1 https://github.com/glimpse-project/glimpse-training-data
git clone --depth=1 https://github.com/glimpse-project/glimpse-models
```
*Note: We recommend you clone the later repositories with --depth=1 since they
contain large resources whose history is rarely required so you can save a lot
of time/bandwidth this way*

Once cloned, please check the respective README.md files in each repo for the
latest instructions but it's expected you will need to also run:

```
cd glimpse-training-data
./unpack.sh
cd blender
./install-addons.sh
```
*(to decompress the CMU motion capture data that we use in our rendering pipeline
and to configure Blender with all the required addons for rendering via
blender/glimpse-cli.py)*

and in the glimpse-models repo run:

```
cd glimpse-models
./unpack.sh
```
*(to decompress the decision trees)*

# Environment variables

Further instructions assume the following environment variables are set

`GLIMPSE_ASSETS_ROOT` is set to an absolute path containing all the data from the glimpse-models repository above. For the `glimpse_viewer` application, the [UI font file](src/Roboto-Medium.ttf) is also expected. This path can also contain recordings and motion capture targets, in `ViewerRecording` and `Targets` directories, respectively.

# Building

Currently we support building and running Glimpse on Linux, OSX and
cross-compiling for Android and iOS. If someone wants to help port to Windows,
that would be greatly appreciated and probably wouldn't be too tricky.

We're using [Meson](https://mesonbuild.com) and [Ninja](https://ninja-build.org/)
for building. If you don't already have Meson, it can typically be installed by
running:
```
pip3 install --user --upgrade meson
```

For cross-compiling to Android or iOS you currently need to use [this branch of
meson](https://github.com/glimpse-project/meson) which e.g. knows not to use
shared library versioning on Android:

```
pip3 install --user --upgrade git+https://github.com/glimpse-project/meson
```
The version should have `glimpse-devX` in the suffix like:
```
$ meson --version
0.46.0.glimpse-dev2
```

## Debug

By default Meson will compile without optimizations and with debug symbols:

```
mkdir build-debug
cd build-debug
meson ..
ninja
```

## Release

An optimized build can be compiled as follows:
```
mkdir build-release
cd build-release
CFLAGS="-march=native -mtune=native" CXXFLAGS="-march=native -mtune=native" meson --buildtype=release ..
ninja
```

# Building for Android

We've only tested cross-compiling with NDK r16 and have certainly had problems
with earlier versions so would strongly recommend using a version >= r16.

For ease of integration with Meson we create a standalone toolchain like so:

```
$ANDROID_NDK_HOME/build/tools/make_standalone_toolchain.py --install-dir ~/local/android-arm-toolchain-24 --arch arm --api 24 --stl libc++
export PATH=~/local/android-arm-toolchain-24/bin:$PATH
```
*Note: we can't build for arm64 when building the libglimpse-unity-plugin.so since Unity doesn't natively support arm64 on Android*
*Note: while building for 32bit arm we have to use api level >= 24 otherwise we hit build issues with -D_FILE_OFFSET_BITS=64 usage*


Make sure you have cloned the `glimpse` branch of Meson from
[here](https://github.com/glimpse-project/meson), since upstream Meson isn't
yet aware that Android lacks support for shared library versioning.

The version should have `glimpse` in the suffix like:
```
$ meson --version
0.45.0.glimpse-dev1
```

If not, then it can be installed like:
```
pip3 install --user --upgrade git+https://github.com/glimpse-project/meson
```


Then to compile Glimpse:
```
mkdir build-android-debug
cd build-android-debug
meson --cross-file ../android-armeabi-v7a-cross-file.txt --buildtype=debug ..
ninja
```

or release:
```
mkdir build-android-release
cd build-android-release
meson --cross-file ../android-armeabi-v7a-cross-file.txt --buildtype=release ..
ninja
```

## Common issues

* Android commands fail

Make sure the platform tools are in the path and make sure the build-tools and platform are installed. They can be installed with the following commands:
```
sdkmanager "platforms;android-23"
sdkmanager "build-tools;26.0.2"
```

* Java commands fail

Make sure that Java SDK 8 is installed and not a more recent version.

* debug.keystore not found when creating APKs

This will be created by Android Studio, but can be created manually with the following command:
```
keytool -genkey -v -keystore ~/.android/debug.keystore -alias androiddebugkey -storepass android -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
```

# Building for OSX

Assuming you use [brew](http://brew.sh/) to install third-party tools then the Glimpse build will
first require:

`brew install pkg-config zlib libpng glfw3 python ninja git`

If you want to use the software with a Kinect device, you will also need libfreenect installed:

`brew install libfreenect`

You will also likely want to add locally installed Python binaries to your PATH, which can be done by adding the following line to ~/.bash_profile:

`export PATH=$HOME/Library/Python/3.6/bin:$PATH`

From this point, building should work the same as on Linux (i.e. install meson, check out the project, create a build directory and issue the meson commands given above).

# Building for iOS

This is mostly the same as building for Android, except XCode needs to be
installed rather than the Android SDK/NDK, and the iOS cross file
(`ios-xcode-arm64-cross-file.txt`) should be used.

When configuring, append the option `--default-library=static`, as dynamic
libraries are not supported on iOS.

Copy a built `glimpse_viewer` binary to
`xcode/GlimpseViewer/GlimpseViewer/Glimpse Viewer` and open
`xcode/GlimpseViewer` in XCode to package and run on an iPhone X.

For the Unity plugin, use `ninja install_plugin` as with other platform builds.
