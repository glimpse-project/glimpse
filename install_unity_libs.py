#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--subdir',
                    help='The name of the Plugins/<arch> subdirectory')
parser.add_argument('--strip',
                    help='The tool for stripping share library symbols')
args = parser.parse_args()

introspect_cmd = os.environ['MESONINTROSPECT'].split()

targets_json = subprocess.check_output(introspect_cmd + [ '--targets' ])

targets = json.loads(targets_json)

if 'DESTDIR' in os.environ:
    dst = os.path.join(os.environ['DESTDIR'], os.environ['MESON_INSTALL_PREFIX'], args.subdir)
else:
    dst = os.path.join(os.environ['MESON_INSTALL_PREFIX'], args.subdir)

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
