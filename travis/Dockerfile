FROM ubuntu:zesty
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
RUN apt-get update -y && apt-get install -y --no-install-recommends --no-install-suggests \
    sudo \
    wget && \
    apt-get clean
RUN echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main" >> /etc/apt/sources.list
RUN wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
RUN apt-get update -y && apt-get install -y --no-install-recommends --no-install-suggests \
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
    libpcl-dev && \
    apt-get clean

# Workaround multiple bugs in Ubuntu's libpcl-dev package :(
# Re: https://bugs.launchpad.net/ubuntu/+source/pcl/+bug/1738902
RUN sed -i 's/pcl_2d-1.8//' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_features-1.8.pc && \
    sed -i 's/Requires:/Requires: eigen3 /g' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_common-1.8.pc

CMD ["/bin/bash"]
