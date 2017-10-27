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
    echo " -n,--native          build for the host machine not Android"
    echo ""
    echo "ENVIRONMENT:"
    echo ""
    echo "  ANDROID_HOME        path to android SDK"
    echo "  ANDROID_NDK_HOME    path to android NDK (version $_MIN_NDK+)"
    echo ""
    exit 1
}

function cmake_build {
    PROJ=$1
    shift

    if test -f $STAGE_DIR/$PROJ.built; then
        echo "$PROJ already built"
    else
        cd $SRC_DIR/$PROJ
        rm -fr $_BUILD_DIR_NAME
        mkdir $_BUILD_DIR_NAME
        cd $_BUILD_DIR_NAME

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
        rm -fr $_BUILD_DIR_NAME

        PATH_SAVE=$PATH

        if test "$_NATIVE_BUILD" = "yes"; then
            SYS_OPTS="target-os=linux"
        else
            export PATH="$STAGE_DIR/toolchain-$_ARCH/bin:$PATH"
            SYS_OPTS="target-os=android"
        fi

	# The out-of-tree build support doesn't work for us and
	# the ./b2 --clean -a doesn't seem to do enough either
	# to trust that everything from previous builds has been
	# cleaned
        git clean -xdf
        git submodule foreach --recursive git clean -xdf
        mkdir $_BUILD_DIR_NAME
	rm -fr $STAGE_DIR/boost
	mkdir -p $STAGE_DIR/boost
        ./bootstrap.sh --without-libraries=python && \
	./b2 headers && \
	./b2 --clean -a && \
        ./b2 install $_J_ARG -q \
            toolset=clang \
            link=$_COMPONENTS \
            define="_FILE_OFFSET_BITS=32" \
            $SYS_OPTS \
            --prefix=$STAGE_DIR/boost \
            --with-system \
            --with-filesystem \
            --with-date_time \
            --with-thread \
            --with-iostreams \
            --with-serialization \
            -sNO_BZIP2=1 && \
        touch $STAGE_DIR/boost.built

        export PATH=$PATH_SAVE
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
    if ! test -d glm; then
        git clone https://github.com/g-truc/glm -b 0.9.8.5 glm
    fi
    if ! test -d pcl; then
        git clone https://github.com/PointCloudLibrary/pcl pcl
    fi

    for zip in TangoSDK_Ikariotikos_C TangoSupport_Ikariotikos_C Tango3DR_Ikariotikos_C
    do
        if ! test -f $zip.zip; then
            wget https://developers.google.com/tango/downloads/$zip.zip
        fi
    done
}


###############################################################################
#
# COMMAND LINE PARSING
#
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


SHORT_OPTS="+hfj:n"
LONG_OPTS="help,fetch-only,jobs,native"
OPTS=$(getopt -n $0 -o $SHORT_OPTS -l $LONG_OPTS -- "$@" || usage)
eval set -- $OPTS
while true; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -f|--fetch-only)
            fetch_only_opt=1
            shift
            ;;
        -j|--jobs)
            _J_ARG="-j$2"
            shift 2
            ;;
        -n|--native)
            _NATIVE_BUILD=yes
            shift
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

set -e
set -x

mkdir -p $1
SRC_DIR=$(get_abs_filename $1)

mkdir -p $2
STAGE_DIR=$(get_abs_filename $2)

echo "SRC_DIR=$SRC_DIR"
echo "STAGE_DIR=$STAGE_DIR"


###############################################################################
#
# PREPARE ENVIRONMENT
#
###############################################################################

_BUILD_NAME=glimpse
_BUILD_DIR_NAME=build-$_BUILD_NAME
_BUILD_TYPE=release
_COMPONENTS=shared
_J_ARG=-j8

_PKG_SUFFIX=${_BUILD_TYPE}_$(date +%F|tr '-' '_')

if test "$_NATIVE_BUILD" != "yes"; then
    _MIN_NDK=15
    _ARCH=arm64
    _ABI=arm64-v8a
    _PLATFORM=21

    # Eventually this will be the only STL supported by the NDK but we can't
    # build with it yet...
    #
    #_MAKE_STANDALONE_STL=libc++
    #_CMAKE_STL=c++_shared

    _MAKE_STANDALONE_STL=gnustl

    # XXX: we actually avoid passing this to cmake because we ended up with
    # libraries depending on a non-existent libgnustl_shared library :/
    _CMAKE_STL=gnustl_shared

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
    _PKG_NAME=third_party_deps_${_ARCH}_${_PKG_SUFFIX}
else
    _ARCH=x64
    _ABI=x64
    _PKG_NAME=third_party_deps_${_ARCH}_${_PKG_SUFFIX}
fi


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
    if ! test -f $SRC_DIR/cmake-$CMAKE_VER-Linux-x86_64.tar.gz; then
        cd $SRC_DIR
        wget https://cmake.org/files/v$CMAKE_MAJOR.$CMAKE_MINOR/cmake-$CMAKE_VER-Linux-x86_64.tar.gz
    fi

    cd $STAGE_DIR
    tar -xvf $SRC_DIR/cmake-$CMAKE_VER-Linux-x86_64.tar.gz
fi

export PATH=$STAGE_DIR/cmake-$CMAKE_VER-Linux-x86_64/bin:$PATH
cmake --version

# We need to cripple pkg-config to not find anything because of cmake's design.
# If we don't cripple pkg-config then projects like dlib end up finding host
# libraries which aren't relevant when cross compiling and lead to link failures.
#
export PKG_CONFIG_LIBDIR=/dummy/foo

# Note we enable ANDROID_DEPRECATED_HEADERS because the new way is not reliably
# supported by the hacks android.toolchain.cmake attempts to use to get cmake
# to understand how headers and libraries are no longer in a single sysroot
# location (e.g. it didn't work for opencv builds). The issue with the new
# 'unified headers' is that cmake can only be told about one sysroot where
# headers are found but it still needs to find the platform libraries organised
# in per-platform directories. Using the deprecated headers means we can have
# single sysroot which is much simpler from the pov of cmake configuration.
#

# XXX: for now we avoid passing ANDROID_STL=gnustl_shared since we were ending
# up with libraries depending on a non-existent libgnustl_shared library...
#    -DANDROID_STL=$_CMAKE_STL
if test "$_NATIVE_BUILD" != "yes"; then
    ANDROID_CMAKE_ARGS="\
        -DANDROID_DEPRECATED_HEADERS=ON \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_NATIVE_API_LEVEL=$_PLATFORM \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
    echo "Android Build"
else
    echo "Native Build"
fi

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
if test "$_NATIVE_BUILD" != "yes"; then
    CMAKE_RPATH_ARG=-Wl,-rpath-link=$ANDROID_NDK_HOME/platforms/android-$_PLATFORM/arch-$_ARCH/usr/lib
fi


###############################################################################
#
# FETCH
#
###############################################################################

fetch_all

if test -n "$fetch_only_opt"; then
    exit 0
fi


###############################################################################
#
# BUILD AND STAGE BINARIES BEFORE PACKAGING
#
###############################################################################

cmake_build dlib
cmake_build opencv -DWITH_TBB=ON -DBUILD_SHARED_LIBS=ON

# We rely on distro packages of libpng for native builds
if test "$_NATIVE_BUILD" != "yes"; then
    cmake_build libpng -Dld-version-script=OFF -DPNG_ARM_NEON=on
fi

cmake_build flann -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF -DBUILD_TESTS=OFF -DUSE_OPENMP=OFF
cmake_build eigen
cmake_build glm
build_boost

cmake_build pcl -DCMAKE_FIND_ROOT_PATH="$STAGE_DIR/eigen;$STAGE_DIR/flann;$STAGE_DIR/boost;$STAGE_DIR/qhull;$STAGE_DIR/libpng" -DWITH_OPENGL=OFF -DWITH_PCAP=OFF -DWITH_OPENNI2=OFF -DHAVE_MM_MALLOC_EXITCODE="FAILED_TO_RUN" -DHAVE_POSIX_MEMALIGN=0

cd $STAGE_DIR

if ! test -d lib_tango_client_api; then
    unzip $SRC_DIR/TangoSDK_Ikariotikos_C.zip
fi
if ! test -d lib_tango_support_api; then
    unzip $SRC_DIR/TangoSupport_Ikariotikos_C.zip
fi
if ! test -d lib_tango_3d_reconstruction_api; then
    unzip $SRC_DIR/Tango3DR_Ikariotikos_C.zip
fi


###############################################################################
#
# PACKAGE
#
###############################################################################

rm -fr $_PKG_NAME

# Package things differently depending on whether it's a native or Android
# build because cmake build layouts are inconsistent across projects and it's
# inconvenient to have <project>/lib/<abi>/ heirachy for native builds
# considering the need to set LD_LIBRARY_PATH to find the libraries at runtime
#
if test "$_NATIVE_BUILD" != "yes"; then
    for i in dlib libpng flann boost pcl
    do
        mkdir -p $STAGE_DIR/$_PKG_NAME/$i/lib/
        cp -a $STAGE_DIR/$i/lib $STAGE_DIR/$_PKG_NAME/$i/lib/$_ABI
    done

    mkdir -p $STAGE_DIR/$_PKG_NAME/dlib/lib/
    cp -a $STAGE_DIR/dlib/lib $STAGE_DIR/$_PKG_NAME/dlib/lib/$_ABI

    mkdir -p $STAGE_DIR/$_PKG_NAME/opencv/lib
    cp -a $STAGE_DIR/opencv/sdk/native/libs/$_ABI $STAGE_DIR/$_PKG_NAME/opencv/lib/$_ABI

    for i in dlib libpng flann boost pcl eigen glm
    do
        mkdir -p $STAGE_DIR/$_PKG_NAME/$i
        cp -a $STAGE_DIR/$i/include $STAGE_DIR/$_PKG_NAME/$i
    done

    cp -a $STAGE_DIR/opencv/sdk/native/jni/include $STAGE_DIR/$_PKG_NAME/opencv
    cp -a $STAGE_DIR/lib_tango_* $STAGE_DIR/$_PKG_NAME
else
    mkdir -p $STAGE_DIR/$_PKG_NAME/lib
    for i in flann boost pcl
    do
        cp -a $STAGE_DIR/$i/lib/lib* $STAGE_DIR/$_PKG_NAME/lib
    done

    cp -a $STAGE_DIR/dlib/lib64/lib* $STAGE_DIR/$_PKG_NAME/lib
    #cp -a $STAGE_DIR/libpng/lib64/lib* $STAGE_DIR/$_PKG_NAME/lib
    cp -a $STAGE_DIR/opencv/lib/lib* $STAGE_DIR/$_PKG_NAME/lib

    mkdir -p $STAGE_DIR/$_PKG_NAME/include
    for i in dlib flann boost glm
    do
        cp -a $STAGE_DIR/$i/include/* $STAGE_DIR/$_PKG_NAME/include
    done

    cp -a $STAGE_DIR/eigen/include/eigen3/Eigen $STAGE_DIR/$_PKG_NAME/include
    cp -a $STAGE_DIR/pcl/include/pcl-1.8/* $STAGE_DIR/$_PKG_NAME/include
    cp -a $STAGE_DIR/opencv/include/* $STAGE_DIR/$_PKG_NAME/include
    cp -a $STAGE_DIR/lib_tango_*/include/* $STAGE_DIR/$_PKG_NAME/include
fi

cd $STAGE_DIR
tar -cjf $_PKG_NAME.tar.bz2 $_PKG_NAME
