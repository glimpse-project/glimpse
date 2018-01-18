#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--plugins-subdir',
                    help='The name of the subdirectory underneath Assets/Plugins/ to install libraries')
parser.add_argument('--android-ndk-arch',
                    help='The arch used to find a copy of libc++_shared.so in the Android NDK (e.g. armeabi-v7a, arm64-v8a, x86_64)')
parser.add_argument('--strip',
                    help='The tool for stripping share library symbols',
                    default='strip')
args = parser.parse_args()

introspect_cmd = os.environ['MESONINTROSPECT'].split()

targets_json = subprocess.check_output(introspect_cmd + [ '--targets' ])

if args.android_ndk_arch:
    if 'ANDROID_NDK_HOME' not in os.environ:
        print("")
        print("ERROR: $ANDROID_NDK_HOME not set so install script can't find libc++_shared.so")
        print("")
        sys.exit(1)

    ndk_path = os.environ['ANDROID_NDK_HOME']
    libcxx_shared_path = os.path.join(ndk_path, 'sources/cxx-stl/llvm-libc++/libs',
                                      args.android_ndk_arch, 'libc++_shared.so')

    print("Installing libc++_shared.so from %s" % libcxx_shared_path)

targets = json.loads(targets_json)

if 'DESTDIR' in os.environ:
    dst = os.path.join(os.environ['DESTDIR'], os.environ['MESON_INSTALL_PREFIX'], args.plugins_subdir)
else:
    dst = os.path.join(os.environ['MESON_INSTALL_PREFIX'], args.plugins_subdir)

print("DST = " + dst)

os.makedirs(dst, exist_ok=True)

libs_blacklist = {
    'flann' # We only need libflann_cpp.so
}

for target in targets:
    if target['type'] == 'shared library' and target['name'] not in libs_blacklist:
        basename = os.path.basename(target['filename'])
        parts = basename.split('.')

        install_cmd = [ 'install', '-s' , '--strip-program', args.strip,
                os.path.join(os.environ['MESON_BUILD_ROOT'], target['filename']), 
                os.path.join(dst, parts[0] + '.so')]
        print(" ".join(install_cmd))
        subprocess.check_call(install_cmd)

        if len(parts) > 2:
            os.chdir(dst)
            ln_cmd = [ 'ln', '-s', parts[0] + '.so', parts[0] + '.so.' + parts[2] ]
            print(" ".join(ln_cmd))
            subprocess.check_call(ln_cmd)

if args.android_ndk_arch:
    install_cmd = [ 'install', '-s' , '--strip-program', args.strip,
                    libcxx_shared_path, dst]
    print(" ".join(install_cmd))
    subprocess.check_call(install_cmd)


os.chdir(dst)
chrpath_cmd = [ 'chrpath', '-r', '$ORIGIN', 'libglimpse-unity-plugin.so' ]
print(" ".join(chrpath_cmd))
subprocess.check_call(chrpath_cmd)
