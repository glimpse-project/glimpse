#!/bin/bash

set -e
set -x

python3 --version
pip3 --version
ninja --version

# Avoid virtualenv since Meson fails with:
# ERROR: Multiple producers for Ninja target
# ../glimpse-py3-env/bin/python3". Please rename your targets.

#sudo pip3 install virtualenv
#
#virtualenv glimpse-py3-env
#source glimpse-py3-env/bin/activate

python --version
pip3 install git+https://github.com/glimpse-project/meson
export PATH=$HOME/.local/bin:$PATH

export CC=clang-5.0 CXX=clang++-5.0

mkdir build
cd build

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

deactivate
