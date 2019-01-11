#!/bin/sh

if ! test -f windows-sdk-10.0.17134.0.tar.bz2; then
    echo "windows-sdk-10.0.17134.0.tar.bz2 should be copied to travis/ directory first"
    exit 1
fi
sudo docker build -t rib1/glimpse-travis .
