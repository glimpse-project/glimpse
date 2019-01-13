FROM ubuntu:bionic
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    sudo \
    unzip \
    gnupg2 \
    wget

RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main" >> /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-6.0 main" >> /etc/apt/sources.list
RUN wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    unzip \
    build-essential \
    ninja-build \
    clang-6.0 \
    lld-6.0 \
    clang-8 \
    lld-8 \
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

RUN apt-get clean

# Workaround multiple bugs in Ubuntu's libpcl-dev package :(
# Re: https://bugs.launchpad.net/ubuntu/+source/pcl/+bug/1738902
RUN sed -i 's/pcl_2d-1.8//' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_features-1.8.pc && \
    sed -i 's/Requires:/Requires: eigen3 /g' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_common-1.8.pc

# Installs Android NDK and creates the toolchain
ENV ANDROID_NDK_VERSION r18b
ENV ANDROID_TARGET_API 28
ENV ANDROID_NDK_HOME /android-ndk-${ANDROID_NDK_VERSION}
ENV ANDROID_NDK_FILENAME android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip
ENV ANDROID_TOOLCHAIN /android-arm-toolchain-${ANDROID_TARGET_API}
ENV PATH ${ANDROID_TOOLCHAIN}/bin:$PATH
RUN echo "Downloading android-ndk" >&2 && \
    wget --no-check-certificate --no-verbose https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME} && \
    echo "Extracting android-ndk" >&2 && \
    unzip ${ANDROID_NDK_FILENAME} && \
    mkdir -p ${ANDROID_TOOLCHAIN} && \
    ${ANDROID_NDK_HOME}/build/tools/make_standalone_toolchain.py \
        --force \
        --install-dir ${ANDROID_TOOLCHAIN} \
        --arch arm \
        --api ${ANDROID_TARGET_API} \
        --stl libc++ && \
    rm ${ANDROID_NDK_FILENAME}

# Installs Android SDK
ENV ANDROID_BUILD_TOOLS 28.0.3
ENV ANDROID_SDK_VERSION 4333796
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
    "platforms;android-${ANDROID_TARGET_API}" \
    "build-tools;${ANDROID_BUILD_TOOLS}" \
    "platform-tools"

ADD windows-sdk /windows-sdk

CMD ["/bin/bash"]
