#!/bin/sh

if ! test -d windows-sdk; then
    echo "Missing required windows-sdk directory (build with ../windows-sdk-build.py"
fi

# Fix up absolute paths in the Windows SDK:
sed -i -r "s|sdk_path = .+|sdk_path = '/windows-sdk'|g" windows-sdk/meson/windows-x64-debug-cross-file.txt
sed -i -r "s|sdk_path = .+|sdk_path = '/windows-sdk'|g" windows-sdk/meson/windows-x64-release-cross-file.txt

sudo docker build -t rib1/glimpse-travis .
