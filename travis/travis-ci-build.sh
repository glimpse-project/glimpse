#!/bin/bash

set -e
set -x

python3 --version
pip3 --version
ninja --version

pip3 install git+https://github.com/glimpse-project/meson
export PATH=$HOME/.local/bin:$PATH

mkdir build
cd build

# Udates the meson -cross-file.txt paths
../windows-sdk-build.py --out /windows-sdk

# Have had builds fail just because Meson hasn't been able to download
# subproject tarballs, so we allow configurations to fail and back off
# for five seconds before testing that configuration succeeds (at
# which point subprojects should have been downloaded and unpacked)
set +e
for i in 1 2 3
do
    meson .. --errorlogs $CONFIG_OPTS || sleep 5
    if test -f build.ninja; then
        break
    fi
done
set -e

ninja
