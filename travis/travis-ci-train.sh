#!/bin/bash

set -e
set -x

if ! test -d rendered-training-data/generated; then
    echo "Training data expected under rendered-training-data/generated"
    exit 1
fi

find rendered-training-data

python3 --version
pip3 --version
ninja --version

sudo pip3 install virtualenv

virtualenv glimpse-py3-env
source glimpse-py3-env/bin/activate

python --version
pip install meson

# Actually we rely on the cached, rendered, training images so we don't
# currently need the training data repo here....
#
#git clone --depth=1 https://github.com/glimpse-project/glimpse-training-data
#pushd glimpse-training-data
#    ./unpack.sh
#popd

#git clone --depth=1 https://github.com/glimpse-project/glimpse-models
#pushd glimpse-models
#    ./unpack.sh
#popd

export CC=clang-5.0 CXX=clang++-5.0

mkdir build
pushd build

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

popd

export PATH=$PWD/src:$PATH
export PATH=$PWD/build:$PATH

image-pre-processor rendered-training-data/generated/test-render \
    rendered-training-data/pre-processed/test-render

indexer.py rendered-training-data/pre-processed/test-render

train_rdt rendered-training-data/pre-processed/test-render full full.rdt -d 3 -p 1000 -t 25 -c 1000

deactivate
