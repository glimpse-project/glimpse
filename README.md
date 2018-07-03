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

![](https://raw.githubusercontent.com/wiki/glimpse-project/glimpse/images/screenshot-2017-12-07.png)

The early direction of this project has been to implement some of the
capabilities demoed in our
['Glimpse — a sneak peek into your creative self'](https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6)
teaser video.

The current focus is on skeletal tracking and on reproducing the capabilities
of Microsoft's Kinect based skeletal tracking, but using mobile phones instead
of the Kinect sensor.

See this paper on [Real-Time Human Pose Recognition in Parts from Single Depth
Images](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf)
before delving into the code for more details on our initial approach.

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

# Environment variable

Further instructions assume the following environment variables are set

`GLIMPSE_TRAINING_DATA` is set to the absolute path for the
glimpse-training-data repository cloned above.

`GLIMPSE_MODELS` is set to the absolute path for the glimpse-models repository
above.

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

For cross-compiling to Android you currently need to use [this branch of
meson](https://github.com/glimpse-project/meson) which knows not to use shared
library versioning on Android:

```
pip3 install --user --upgrade git+https://github.com/glimpse-project/meson
```
The version should have `glimpse` in the suffix like:
```
$ meson --version
0.45.0.glimpse-dev1
```

For cross-compiling to iOS, you may need to use [this branch of
meson](https://github.com/glimpse-project/meson/tree/wip/rib/ios):

```
pip3 install --user --upgrade git+https://github.com/glimpse-project/meson@wip/rib/ios
```
The version should have `glimpse` in the suffix like:
```
$ meson --version
0.46.0.glimpse-dev1
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

Assuming you use `brew` to install third-party tools then the Glimpse build will
first require:

`brew install pkg-config libpng glfw3`

# Building for iOS

This is mostly the same as building for Android, except XCode needs to be
installed rather than the Android SDK/NDK, and the iOS cross file
(`ios-xcode-arm64-cross-file.txt`) should be used.

When configuring, append the option `--default-library=static`, as dynamic
libraries are not supported on iOS.

Building will produce binaries that can be imported into an XCode project and
run on device.
