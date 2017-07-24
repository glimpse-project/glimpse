#!/bin/bash

# This iterates through a directory of rendered images and compares sequential
# images with the perceptualdiff tool to determine whether images are similar
# and record whether images can be ignored for the purposes of training our
# randomized decision forest.
#
# The script only records it's results to stdout so the output should be saved,
# and possibly reviewed by hand before deciding to finally delete the redundant
# images, e.g. via:
#
#  for file in `grep '\.png' log|grep remove|cut -d' ' -f2`
#  do
#      rm $file
#  done


if test $# != 1; then
    echo "Usage $0 IMAGE_DIRECTORY"
    exit 1
fi

for dir in `find $1 -type d -iname '*_*'`
do
    echo "$dir:"

    prev=""

    pushd $dir &> /dev/null
    for f in `find ./ -iname '*.png'`
    do
        if test "$prev" != ""; then
            if perceptualdiff --fov 83 --threshold 200 $prev $f &> /dev/null; then
                echo "remove $dir/$f: $prev and $f similar"
            else
                #echo "$dir: $prev and $f differ"
                prev="$f"
            fi
        else
            prev="$f"
        fi
    done
    popd &> /dev/null
done
