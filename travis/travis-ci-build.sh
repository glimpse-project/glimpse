#!/bin/bash

set -e
set -x

python3 --version
pip3 --version
ninja --version

sudo pip3 install virtualenv

virtualenv glimpse-py3-env
source glimpse-py3-env/bin/activate

python --version
pip install meson

if test "$ANDROID_BUILD" = "1"; then
    export ANDROID_NDK_HOME=$PWD/android-ndk-r16b
    if ! test -d $ANDROID_NDK_HOME/bin; then
        wget https://dl.google.com/android/repository/android-ndk-r16b-linux-x86_64.zip
        unzip -q android-ndk-r16b-linux-x86_64.zip
        $ANDROID_NDK_HOME/build/tools/make_standalone_toolchain.py --force --install-dir ./android-arm64-toolchain-21 --arch arm64 --api 21 --stl libc++;
    fi

    export PATH=$PWD/android-arm64-toolchain-21/bin:$PATH
fi

export CC=clang-5.0 CXX=clang++-5.0

mkdir build
cd build

# Have had builds fail just because Meson hasn't been able to download
# subproject tarballs, so we allow configurations to fail and back off
# for five seconds before testing that configuration succeeds (at
# which point subprojects should have been downloaded and unpacked)
set +e
for i in 1 2 3
do
    meson .. --errorlogs --warnlevel 3 $CONFIG_OPTS || sleep 5
    if test -f build.ninja; then
        break
    fi
done
set -e

ninja -v

deactivate
