#!/bin/bash

set -e
set -x

if test -d rendered-training-data/generated; then
    exit 0
fi

python3 --version
pip3 --version
ninja --version

pip3 install meson
export PATH=$HOME/.local/bin:$PATH

# Even though it's an out-of-date version we install Blender via apt-get
# as an easy way of installing dependencies
sudo apt-get install blender -y --no-install-recommends --no-install-suggests

if ! test -f blender-2.79-linux-glibc219-x86_64/blender; then
    wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
    tar -xf blender-2.79-linux-glibc219-x86_64.tar.bz2;
fi
export PATH=$PWD/blender-2.79-linux-glibc219-x86_64:$PATH
blender --version

pip3 install numpy
git clone --depth=1 https://github.com/glimpse-project/glimpse-training-data
pushd glimpse-training-data
    ./unpack.sh
popd
pushd glimpse-training-data/blender
    ./install-addons.sh
popd

export PATH=$PWD/glimpse-training-data/blender:$PATH

mkdir -p rendered-training-data

glimpse-cli.py \
    --start 25 \
    --end 26 \
    --preload \
    glimpse-training-data

glimpse-cli.py \
    --start 25 \
    --end 26 \
    --dest rendered-training-data \
    --name test-render \
    glimpse-training-data

find rendered-training-data
