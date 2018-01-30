#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys


os.chdir(os.environ['MESON_BUILD_ROOT'])

parser = argparse.ArgumentParser()
parser.add_argument('--buildtype',
                    help="The Meson configured buildtype")
parser.add_argument('--viewer',
                    help='Absolute path in which to install Glimpse Viewer and dependent libraries')
parser.add_argument('--unity-project',
                    help='Absolute path to a Unity project where we will install the Glimpse plugin')
parser.add_argument('--plugin-libdir',
                    help='Relative path under Unity project where libglimpse-unity-plugin.so will be copied')
parser.add_argument('--plugin-jardir',
                    help='Relative path under Unity project where GlimpsePlugin.jar will be copied (Android only)')
parser.add_argument('--tango-libs',
                    help='Location of Tango libraries (Android only)')
parser.add_argument('--android-ndk-arch',
                    help='The arch used to find a copy of libc++_shared.so in the Android NDK (e.g. armeabi-v7a, arm64-v8a, x86_64)')
parser.add_argument('--strip',
                    help='The tool for stripping share library symbols',
                    default='strip')
args = parser.parse_args()

introspect_cmd = os.environ['MESONINTROSPECT'].split()
targets_json = subprocess.check_output(introspect_cmd + [ '--targets' ])

targets = json.loads(targets_json)
libs_blacklist = {
    'flann' # We only need libflann_cpp.so
}

if args.buildtype == "release":
    lib_strip_args = [ '-s', '--strip-program', args.strip ]
else:
    lib_strip_args = []

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


def do_install(src, dst, extra_args=[]):
    install_cmd = [
            'install',
            ] + extra_args + [ src, dst ]

    print(" ".join(install_cmd))
    subprocess.check_call(install_cmd)


def install_shared_lib_targets(dst, blacklist):
    for target in targets:
        if target['name'] in blacklist:
            continue

        if target['type'] == 'shared library':
            basename = os.path.basename(target['filename'])
            parts = basename.split('.')

            do_install(os.path.join(os.environ['MESON_BUILD_ROOT'], target['filename']),
                       os.path.join(dst, parts[0] + '.so'), lib_strip_args)

            if len(parts) > 2:
                os.chdir(dst)
                ln_cmd = [ 'ln', '-sf', parts[0] + '.so', parts[0] + '.so.' + parts[2] ]
                print(" ".join(ln_cmd))
                subprocess.check_call(ln_cmd)


def install_jar_targets(dst):
    for target in targets:

        if target['filename'][-4:] == '.jar':
            do_install(os.path.join(os.environ['MESON_BUILD_ROOT'], target['filename']),
                       os.path.join(dst, target['filename']))


def install_unity_plugin(unity_project):
    unity_plugin_libs_dir = os.path.join(unity_project, args.plugin_libdir)
    os.makedirs(unity_plugin_libs_dir, exist_ok=True)
    install_shared_lib_targets(unity_plugin_libs_dir,
                               libs_blacklist.union({'glimpse_viewer_android'}))

    if args.android_ndk_arch:
        unity_plugin_jar_dir = os.path.join(unity_project, args.plugin_jardir)
        os.makedirs(unity_plugin_jar_dir, exist_ok=True)
        install_jar_targets(unity_plugin_jar_dir)

        do_install(libcxx_shared_path, unity_plugin_libs_dir)

        if args.tango_libs:
            tango_libs = [ 'libtango_support_api.so' ]
            for lib in tango_libs:
                do_install(os.path.join(args.tango_libs, 'lib', args.android_ndk_arch, lib),
                           unity_plugin_libs_dir)

    # To avoid needing to set LD_LIBRARY_PATH for the plugin to load its
    # dependencies we update the RPATH to look in the same directory as
    # the plugin itself...
    os.chdir(unity_plugin_libs_dir)
    chrpath_cmd = [ 'chrpath', '-r', '$ORIGIN', 'libglimpse-unity-plugin.so' ]
    print(" ".join(chrpath_cmd))
    subprocess.check_call(chrpath_cmd)

def install_viewer(dst):
    os.makedirs(dst, exist_ok=True)
    install_shared_lib_targets(dst,
                               libs_blacklist.union({'glimpse-unity-plugin'}))

    if args.android_ndk_arch:
        do_install(libcxx_shared_path, dst);

        if args.tango_libs:
            tango_libs = [ 'libtango_support_api.so' ]
            for lib in tango_libs:
                do_install(os.path.join(args.tango_libs, 'lib',
                           args.android_ndk_arch, lib), dst)

if args.unity_project:
    install_unity_plugin(args.unity_project)

if args.viewer:
    install_viewer(args.viewer)
