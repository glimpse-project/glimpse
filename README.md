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

The current focus is on skeletal tracking and on reproducing the capabilities
of Microsoft's Kinect based skeletal tracking, but using mobile phones instead
of the Kinect sensor.

See this paper on [Real-Time Human Pose Recognition in Parts from Single Depth
Images](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf)
before delving into the code for more details on our initial approach.


# Building

So far we've been developing on Linux and we're also targeting cross-compilation
to Android. If someone wants to help port to Windows or OSX, that
would be greatly appreciated and probably wouldn't be too tricky.

We're using [Meson](https://mesonbuild.com) and [Ninja](https://ninja-build.org/)
for building. If you don't already have Meson, it can typically be installed by
running:
```
pip3 install --user meson
```

*Make sure you have Meson >= 0.44.0 if you want to try cross-compiling Glimpse:*
```
pip3 install --user meson --upgrade
meson --version
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
CFLAGS="-march=native -mtune=native" CXXFLAGS="-march=native -mtune=native" meson.py .. --buildtype=release
ninja
```

# Building for Android

We've only tested cross-compiling with NDK r16 and have certainly had problems
with earlier versions so would strongly recommend using a version >= r16.

For ease of integration with Meson we create a standalone toolchain like so:

```
$ANDROID_NDK_HOME/build/tools/make_standalone_toolchain.py --install-dir ~/local/android-arm64-toolchain-21 --arch arm64 --api 21 --stl libc++
export PATH=~/local/android-arm64-toolchain-21:$PATH
```
*(We've only been targeting arm64 devices so far)*

Then to compile Glimpse:
```
mkdir build-android-debug
cd build-android-debug
meson.py --cross-file ../android-arm64-cross-file.txt .. -Ddlib:support_gui=false
ninja
```

or release:
```
mkdir build-android-release
cd build-android-release
meson.py --cross-file ../android-arm64-cross-file.txt .. -Ddlib:support_gui=false --buildtype=release
ninja
```
