#!/bin/bash

# Copyright (c) 2017 Glimp IP Ltd
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

# The intial aim of this script is to simplify configuring Blender
# non-interactively as part of continuous integration
#
# TODO: rewrite this script in Python to run on Windows too
#

blender_version=`blender --version|head -1|cut -d' ' -f2`

if test $blender_version != "2.79"; then
    echo "Only support Blender 2.79"
    exit 1
fi

if test -d blendertools; then
    echo "WARNING: installing makehuman addons from pre-existing ./blendertools/ directory"
    echo ""
else
    tools_ver="1.1.0"
    tools_zip=blendertools-$tools_ver-all.zip
    if ! test -f $tools_zip; then
        wget http://download.tuxfamily.org/makehuman/releases/$tools_ver/$tools_zip
    fi
    unzip $tools_zip
fi

blender_env=`blender -b -P enable-addons.py -- --info|grep BLENDER_USER_`
eval $blender_env

echo "Blender paths:"
echo "BLENDER_USER_CONFIG=$BLENDER_USER_CONFIG"
echo "BLENDER_USER_SCRIPTS=$BLENDER_USER_SCRIPTS"
echo ""

if test -z "$BLENDER_USER_CONFIG"; then
    echo "Failed to determine BLENDER_USER_CONFIG directory"
    exit 1
fi

mkdir -p $BLENDER_USER_CONFIG/addons

for addon in makeclothes maketarget makewalk
do
    if test -d $BLENDER_USER_CONFIG/addons/$addon; then
        echo "WARNING: Not overwritting pre-installed $addon addon"
        echo "(delete $BLENDER_USER_CONFIG/addons/$addon and re run script if necessary)"
    else
        echo "Installing (copying) $addon addon to $BLENDER_USER_CONFIG/addons/$addon"
        cp -a blendertools/$addon $BLENDER_USER_CONFIG/addons
    fi
done

echo ""

#for addon in glimpse_data_generator mesh_paint_rig
#do
#    if test -d $BLENDER_USER_CONFIG/addons/$addon -o \
#            -L $BLENDER_USER_CONFIG/addons/$addon; then
#        echo "WARNING: Not overwritting pre-installed $addon addon"
#    else
#        echo "Installing (linking) $addon addon to $BLENDER_USER_CONFIG/addons/$addon"
#        addon_dir=$PWD/addons/$addon
#        pushd $BLENDER_USER_CONFIG/addons &> /dev/null
#            ln -s $addon_dir
#        popd
#    fi
#done

blender -b -P enable-addons.py
