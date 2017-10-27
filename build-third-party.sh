#!/bin/bash
#
# Copyright (c) 2017 Kwamecorp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This script automates the process of building all our third party android
# dependencies
# 

set -e
set -x

_MIN_NDK=15
_ARCH=arm64
_ABI=arm64-v8a
_PLATFORM=21
_COMPONENTS=shared
_TYPE=release

_PKG_SUFFIX=$_TYPE
_PKG_NAME=third_party_deps_$_PKG_SUFFIX

# So far it only works to use gnustl_shared...
#_MAKE_STANDALONE_STL=libc++
#_CMAKE_STL=c++_shared
_MAKE_STANDALONE_STL=gnustl
_CMAKE_STL=gnustl_shared

_J_ARG=-j8

BUILD_NAME=glimpse
BUILD_DIR_NAME=build-$BUILD_NAME

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


function get_abs_filename {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

function usage {
    set +x
    echo "Usage: $0 [options] src_dir stage_dir"
    echo ""
    echo " -h,--help            print help and exit"
    echo " -f,--fetch-only      only fetch source code and exit"
    echo " -j,--jobs            allow N jobs at once (passed to make)"
    echo ""
    echo "ENVIRONMENT:"
    echo ""
    echo "  ANDROID_HOME        path to android SDK"
    echo "  ANDROID_NDK_HOME    path to android NDK (version $_MIN_NDK+)"
    exit 1
}

eval set -- $(getopt -n $0 -o "+hfj:" -l "help,fetch-only,jobs" -- "$@")
while true; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -f|--fetch-only)
            fetch_opt=1
            shift
            ;;
        -j|--jobs)
            _J_ARG="-j$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

if test $# -ne 2; then
    usage
fi

if ! test -d $1; then
    echo "source directory $1 doesn't exist"
    usage
fi
SRC_DIR=$(get_abs_filename $1)

mkdir -p $2
STAGE_DIR=$(get_abs_filename $2)

echo "SRC_DIR=$SRC_DIR"
echo "STAGE_DIR=$STAGE_DIR"

if test -z $ANDROID_HOME; then
    echo "ANDROID_HOME should point to an installation of the Android SDK"
    exit 1
fi
if test -z $ANDROID_NDK_HOME; then
    echo "ANDROID_NDK_HOME should point to an installation of the Android NDK (version $_MIN_NDK+)"
    exit 1
fi

_NDK_MAJOR_VER=$(cat $ANDROID_NDK_HOME/source.properties | \
    grep Pkg.Revision | \
    cut -d'=' -f2 | \
    awk '{ print $1 }' | \
    cut -d'.' -f1)

if test $_NDK_MAJOR_VER -lt $_MIN_NDK; then
    echo "Expecting to build with NDK version $_MIN_NDK+"
    usage
fi

if ! test -d $STAGE_DIR/toolchain-$_ARCH; then
    $ANDROID_NDK_HOME/build/tools/make_standalone_toolchain.py \
        --arch $_ARCH --api $_PLATFORM --install-dir $STAGE_DIR/toolchain-$_ARCH --stl=$_MAKE_STANDALONE_STL
fi
#export PATH=$STAGE_DIR/toolchain-$_ARCH/bin:$PATH


# We fetch specific cmake binaries because we've seen cmake releases break 
# backwards compatibility in ways that have affected building some of our
# dependencies, and it wastes too much time chasing down those kinds of
# build breakages.
#
# Generally speaking I haven't seen cmake 3.7+ be usable with the Android NDK
# and it looks like others have also hit various problems too, such as:
# https://github.com/android-ndk/ndk/issues/254
#
# Note: at one point I tried using the cmake that's part of the Android SDK
# which has its own android.toolchain.cmake but found it was tied to a different
# version of the NDK than I was trying to use.
#
CMAKE_MAJOR=3
CMAKE_MINOR=6
CMAKE_MICRO=3
CMAKE_MAJOR_VER=$CMAKE_MAJOR.$CMAKE_MINOR
CMAKE_VER=$CMAKE_MAJOR_VER.$CMAKE_MICRO
if ! test -d $STAGE_DIR/cmake-$CMAKE_VER-Linux-x86_64; then
    cd $STAGE_DIR
    wget https://cmake.org/files/v$CMAKE_MAJOR.$CMAKE_MINOR/cmake-$CMAKE_VER-Linux-x86_64.tar.gz
    tar -xvf cmake-$CMAKE_VER-Linux-x86_64.tar.gz
fi

export PATH=$STAGE_DIR/cmake-$CMAKE_VER-Linux-x86_64/bin:$PATH
cmake --version


# We need to cripple pkg-config to not find anything because of cmake's design.
# If we don't cripple pkg-config then projects like dlib end up finding host
# libraries which aren't relevant when cross compiling and lead to link failures.
#
export PKG_CONFIG_LIBDIR=/dummy/foo

#ANDROID_CMAKE_ARGS="-DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_STL=gnustl_shared \
#-DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=21 \
#-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"


# Note we enable ANDROID_DEPRECATED_HEADERS because the new way is not reliably
# supported by the hacks android.toolchain.cmake attempts to use to get cmake
# to understand how headers and libraries are no longer in a single sysroot
# location (e.g. it didn't work for opencv builds). The issue with the new
# 'unified headers' is that cmake can only be told about one sysroot where
# headers are found but it still needs to find the platform libraries organised
# in per-platform directories. Using the deprecated headers means we can have
# single sysroot which is much simpler from the pov of cmake configuration.
#

#    -DANDROID_STL=$_CMAKE_STL \
ANDROID_CMAKE_ARGS="\
    -DANDROID_DEPRECATED_HEADERS=ON \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_NATIVE_API_LEVEL=$_PLATFORM \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"


#
# (ﾉ≧∇≦)ﾉ ﾐ ┸━┸
#
# Really what is the point of cmake!? The amount of time wasted deciphering how
# the android.toolchain.cmake is fighting with and against cmake and all the
# inconsistent ways different projects actually use cmake it's all a fragile
# tower of cards.
#
# The next problem is with projects (e.g. opencv) linking shared libraries
# or executables that depend indirectly on libraries that should be found in
# the ndk toolchain 'sysroot' (a term that's a bit more confusing with the new
# unified headers that we aren't using because they also break our cmake
# builds).
#
# E.g. an executable indirectly depending on zlib won't necessarily have -lz on
# the command line. This is what rpath-link is for but for some reason it's not
# being used by cmake so we have to ensure it's passed ourselves. This is extra
# awkward because it's the full sysroot path for the platform libraries in the
# ndk which we really shouldn't have to care about here :(
#
# I don't know whether to consider this an issue with cmake or
# android.toolchain.cmake for not being complete enough.
#
CMAKE_RPATH_ARG=-Wl,-rpath-link=$ANDROID_NDK_HOME/platforms/android-$_PLATFORM/arch-$_ARCH/usr/lib

function cmake_build {
    PROJ=$1
    shift

    if test -f $STAGE_DIR/$PROJ.built; then
        echo "$PROJ already built"
    else
        cd $SRC_DIR/$PROJ
        rm -fr $BUILD_DIR_NAME
        mkdir $BUILD_DIR_NAME
        cd $BUILD_DIR_NAME

	rm -fr $STAGE_DIR/$PROJ

        #
        # *** Did I ever mention I really really dislike cmake ***
        #
        # For future reference: to enable the sensible behaviour where
        # explicitly specified exe linker flags get used with cmake's
        # try_compile() utility then you have to explicitly turn on the 'NEW'
        # good behaviour with:
        #
        #   -DCMAKE_POLICY_DEFAULT_CMP0056=NEW
        # 
        # See  `cmake --help-policy CMP0056` for more details
        #
        # Without this then it's a major headache to try and use
        # -fsanitize=address if building for Android since the asan runtime
        # library seems to depend on liblog and executables don't link without
        # an explicit -llog (which seems like a toolchain/ndk problem)
        #
        # *** Did I ever mention I really really dislike cmake ***
        #
        #cmake -v --debug-trycompile $ANDROID_CMAKE_ARGS \
        #    -DCMAKE_POLICY_DEFAULT_CMP0056=NEW \
        #    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer -fsanitize=address" \
        #    -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -fsanitize=address" \
        #    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address -llog" \
        #    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address -llog" \
        #    -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address -llog" \
        #    -DCMAKE_C_STANDARD_LIBRARIES="-lm -llog" \
        #    $@ \

        cmake $ANDROID_CMAKE_ARGS \
            -DCMAKE_POLICY_DEFAULT_CMP0056=NEW \
            -DCMAKE_EXE_LINKER_FLAGS="$CMAKE_RPATH_ARG" \
            -DCMAKE_SHARED_LINKER_FLAGS="$CMAKE_RPATH_ARG" \
            $@ \
            -DCMAKE_INSTALL_PREFIX=$STAGE_DIR/$PROJ ../ && \
        make $_J_ARG VERBOSE=1 && \
        cmake --build . --target install && \
        touch $STAGE_DIR/$PROJ.built
    fi
}


function build_boost {
    if test -f $STAGE_DIR/boost.built; then
        echo "boost already built"
    else
        cd $SRC_DIR/boost
        rm -fr $BUILD_DIR_NAME

	# The out-of-tree build support doesn't work for us and
	# the ./b2 --clean -a doesn't seem to do enough either
	# to trust that everything from previous builds has been
	# cleaned
        git clean -xdf
        git submodule foreach --recursive git clean -xdf
        mkdir $BUILD_DIR_NAME
	rm -fr $STAGE_DIR/boost
	mkdir -p $STAGE_DIR/boost
        ./bootstrap.sh --without-libraries=python && \
	./b2 headers && \
	./b2 --clean -a && \
        PATH="$STAGE_DIR/toolchain-$_ARCH/bin:$PATH" ./b2 install $_J_ARG -q \
            toolset=clang \
            link=$_COMPONENTS \
            define="_FILE_OFFSET_BITS=32" \
            target-os=android \
            --prefix=$STAGE_DIR/boost \
            --with-system \
            --with-filesystem \
            --with-date_time \
            --with-thread \
            --with-iostreams \
            --with-serialization \
            -sNO_BZIP2=1 && \
	touch $STAGE_DIR/boost.built
    fi
}


function fetch_all {
    cd $SRC_DIR

    if ! test -d dlib; then
        git clone git@gitlab.com:impossible/dlib.git -b glimpse dlib
    fi
    if ! test -d opencv; then
        git clone https://github.com/opencv/opencv -b master opencv
        git checkout 2c1b4f571123cae115850c49830c217783669270
    fi
    if ! test -d libpng; then
        git clone https://github.com/glennrp/libpng -b libpng16 libpng
    fi
    if ! test -d boost; then
        # NB: if you want to change branch you need to use git submodule --recursive since
        # boost is comprised of a huge number of submodule repos.
        git clone --recursive https://github.com/boostorg/boost.git -b boost-1.61.0 boost

	# The Android NDK's support for _FILE_OFFSET_BITS=64 is broken and Boost makes
	# incorrect assumptions about how it can enable 64 bit offsets on different platforms
	# which conflicts with the NDK. Build error manifests like:
	#
	#  include/c++/4.9.x/cstdio:137:9: error: no member named 'fgetpos' in the global namespace
	#  using ::fgetpos;
	#        ~~^
	#  include/c++/4.9.x/cstdio:139:9: error: no member named 'fsetpos' in the global namespace
	#  using ::fsetpos;
	#
	#  See: https://github.com/android-ndk/ndk/issues/480
	#
        patch -p1 -d boost/libs/filesystem < $SCRIPT_DIR/0001-boost-filesystems-avoid-_FILE_OFFSET_BITS-64-broken-.patch
        patch -p1 -d boost < $SCRIPT_DIR/0001-boost-avoid-versioned-sonames.patch
    fi
    if ! test -d flann; then
        git clone git://github.com/mariusmuja/flann.git
    fi
    if ! test -d eigen; then
        hg clone https://bitbucket.org/eigen/eigen/
    fi
    if ! test -d qhull; then
        # NB: This is a non-upstream branch because the upstream cmake build is totally broken
        git clone https://github.com/jamiesnape/qhull -b cmake
    fi
    if ! test -d pcl; then
        git clone https://github.com/PointCloudLibrary/pcl pcl
    fi
}

fetch_all

if test -n "$fetch_only_opt"; then
    exit 0
fi


cmake_build dlib
cmake_build opencv -DWITH_TBB=ON -DBUILD_SHARED_LIBS=ON
cmake_build libpng -Dld-version-script=OFF -DPNG_ARM_NEON=on
cmake_build flann -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF -DBUILD_TESTS=OFF -DUSE_OPENMP=OFF
cmake_build eigen
build_boost

cmake_build pcl -DCMAKE_FIND_ROOT_PATH="$STAGE_DIR/eigen;$STAGE_DIR/flann;$STAGE_DIR/boost;$STAGE_DIR/qhull;$STAGE_DIR/libpng" -DWITH_OPENGL=OFF -DWITH_PCAP=OFF -DWITH_OPENNI2=OFF -DHAVE_MM_MALLOC_EXITCODE="FAILED_TO_RUN" -DHAVE_POSIX_MEMALIGN=0

cd $STAGE_DIR
rm -fr $_PKG_NAME

for i in dlib libpng flann boost pcl
do
    mkdir -p $STAGE_DIR/$_PKG_NAME/$i/lib/
    cp -a $STAGE_DIR/$i/lib $STAGE_DIR/$_PKG_NAME/$i/lib/$_ABI
done

mkdir -p $STAGE_DIR/$_PKG_NAME/opencv/lib
cp -a $STAGE_DIR/opencv/sdk/native/libs/$_ABI $STAGE_DIR/$_PKG_NAME/opencv/lib/$_ABI

for i in dlib libpng flann boost pcl eigen
do
    mkdir -p $STAGE_DIR/$_PKG_NAME/$i
    cp -a $STAGE_DIR/$i/include $STAGE_DIR/$_PKG_NAME/$i
done

cp -av $STAGE_DIR/opencv/sdk/native/jni/include $STAGE_DIR/$_PKG_NAME/opencv

for zip in TangoSDK_Ikariotikos_C TangoSupport_Ikariotikos_C Tango3DR_Ikariotikos_C
do
    cd $STAGE_DIR
    if ! test -f $zip.zip; then
	wget https://developers.google.com/tango/downloads/$zip.zip
    fi
    cd $STAGE_DIR/$_PKG_NAME
    unzip ../$zip.zip
done

cd $STAGE_DIR
tar -cjf $_PKG_NAME.tar.bz2 $_PKG_NAME
