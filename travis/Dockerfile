FROM ubuntu:artful
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    sudo \
    unzip \
    wget

RUN echo "deb http://apt.llvm.org/artful/ llvm-toolchain-artful-5.0 main" >> /etc/apt/sources.list
RUN wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    unzip \
    build-essential \
    ninja-build \
    clang-5.0 \
    python3-minimal \
    python3-pip \
    python3-setuptools \
    git \
    pkg-config \
    libz-dev \
    libpng-dev \
    libepoxy-dev \
    libfreenect-dev \
    libglfw3-dev \
    libglm-dev \
    libpcl-dev \
    openjdk-8-jdk

# Workaround multiple bugs in Ubuntu's libpcl-dev package :(
# Re: https://bugs.launchpad.net/ubuntu/+source/pcl/+bug/1738902
RUN sed -i 's/pcl_2d-1.8//' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_features-1.8.pc && \
    sed -i 's/Requires:/Requires: eigen3 /g' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_common-1.8.pc

# Installs Android NDK and creates the toolchain
ENV ANDROID_NDK_VERSION r16b
ENV ANDROID_NDK_HOME /android-ndk-${ANDROID_NDK_VERSION}
ENV ANDROID_NDK_FILENAME android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip
ENV ANDROID_TOOLCHAIN /android-arm-toolchain-24
ENV PATH ${ANDROID_TOOLCHAIN}/bin:$PATH
RUN echo "Downloading android-ndk" >&2 && \
    wget --no-check-certificate --no-verbose https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME} && \
    echo "Extracting android-ndk" >&2 && \
    unzip ${ANDROID_NDK_FILENAME} && \
    mkdir -p ${ANDROID_TOOLCHAIN} && \
    ${ANDROID_NDK_HOME}/build/tools/make_standalone_toolchain.py --force --install-dir ${ANDROID_TOOLCHAIN} --arch arm --api 24 --stl libc++ && \
    rm ${ANDROID_NDK_FILENAME}

# Installing Android SDK compat libraries
RUN dpkg --add-architecture i386
RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    libc6:i386 \
    libstdc++6:i386 \
    libgcc1:i386 \
    libncurses5:i386 \
    libz1:i386

# Installs Android SDK
ENV ANDROID_SDK_VERSION 3859397
ENV ANDROID_SDK_FILENAME sdk-tools-linux-${ANDROID_SDK_VERSION}.zip
ENV ANDROID_HOME /android-sdk
ENV PATH ${PATH}:${ANDROID_HOME}/tools:${ANDROID_HOME}/tools/bin:${ANDROID_HOME}/platform-tools

RUN mkdir -p ${ANDROID_HOME} && \
    echo "Downloading android-sdk" >&2 && \
    wget --no-check-certificate --no-verbose https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME} && \
    echo "Extracting android-sdk" >&2 && \
    unzip ${ANDROID_SDK_FILENAME} -d ${ANDROID_HOME} && \
    rm ${ANDROID_SDK_FILENAME}

RUN yes | sdkmanager --licenses
RUN sdkmanager "tools" "platform-tools"
RUN yes | sdkmanager \
    "platforms;android-27" \
    "platforms;android-23" \
    "build-tools;27.0.3"

RUN apt-get clean

CMD ["/bin/bash"]
