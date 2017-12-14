#!/bin/bash

set -e
set -x

python3 --version
pip3 --version

sudo pip3 install virtualenv

virtualenv glimpse-py3-env
source glimpse-py3-env/bin/activate

python --version

# Even though it's an out-of-date version we install Blender via apt-get
# as an easy way of installing dependencies
sudo apt-get install blender -y --no-install-recommends --no-install-suggests

if ! test -f blender-2.79-linux-glibc219-x86_64/blender; then
    wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
    tar -xf blender-2.79-linux-glibc219-x86_64.tar.bz2;
fi
export PATH=$PWD/blender-2.79-linux-glibc219-x86_64:$PATH
blender --version

pip install numpy
git clone --depth=1 https://github.com/glimpse-project/glimpse-training-data
pushd glimpse-training-data
    ./unpack.sh
popd
pushd glimpse-training-data/blender
    ./install-addons.sh
    ./glimpse-cli.py --help
    echo "Trying to run glimpse-cli.py --info ../:"
    ./glimpse-cli.py --info ../|grep -q "75_18: Uncached" && echo "Ran glimpse-cli.py --info OK"
popd

git clone --depth=1 https://github.com/glimpse-project/glimpse-models
pushd glimpse-models
    ./unpack.sh
popd

deactivate
