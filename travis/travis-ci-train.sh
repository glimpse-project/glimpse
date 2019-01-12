#!/bin/bash

set -e
set -x

if ! test -d rendered-training-data/rendered; then
    echo "Training data expected under rendered-training-data/rendered"
    exit 1
fi

find rendered-training-data

python3 --version
pip3 --version
ninja --version

pip3 install git+https://github.com/glimpse-project/meson
export PATH=$HOME/.local/bin:$PATH

git clone --depth=1 https://github.com/glimpse-project/glimpse-training-data

# We rely on the cached, rendered, training images so we don't currently need
# the unpack training data here....
#
#pushd glimpse-training-data
#    ./unpack.sh
#popd

#git clone --depth=1 https://github.com/glimpse-project/glimpse-models
#pushd glimpse-models
#    ./unpack.sh
#popd

export CC=clang-6.0 CXX=clang++-6.0

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

image-pre-processor \
    rendered-training-data/rendered/test-render \
    rendered-training-data/pre-processed/test-render \
    glimpse-training-data/label-maps/2018-11-render-to-2018-08-rdt-map.json \
    --config glimpse-training-data/pre-processor-configs/iphone-x-config.json

glimpse-training-data/glimpse-data-indexer.py rendered-training-data/pre-processed/test-render

JOBS=$(cat<<'EOF'
[
    {
        "index_name": "full",
        "out_file": "full-d2.json",
        "max_depth": 2,
        "n_pixels": 500,
        "n_thresholds": 25,
        "n_uvs": 500,
        "pretty": true
    },
    {
        "index_name": "full",
        "reload": "full-d2.json",
        "out_file": "full-d3.json",
        "max_depth": 3,
        "n_pixels": 500,
        "n_thresholds": 25,
        "n_uvs": 500,
        "pretty": true
    }
]
EOF
)

train_rdt --log-stderr -q "$JOBS" -d rendered-training-data/pre-processed/test-render full full-tree.json
