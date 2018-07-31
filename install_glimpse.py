#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys


if 'MESON_BUILD_ROOT' in os.environ:
    build_dir = os.environ['MESON_BUILD_ROOT']
    os.chdir(os.environ['MESON_BUILD_ROOT'])
else:
    build_dir = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--buildtype',
                    help="The Meson configured buildtype",
                    default='debug')
parser.add_argument('--tango-libs',
                    help='Location of Tango libraries (Android only)')
parser.add_argument('--android-ndk',
                    help='Path to the Android NDK',
                    default=os.getenv("ANDROID_NDK_HOME", ""))
parser.add_argument('--android-ndk-arch',
                    help='The arch used to find a copy of libc++_shared.so in the Android NDK (e.g. armeabi-v7a, arm64-v8a, x86_64)')
parser.add_argument('--strip',
                    help='The tool for stripping shared library symbols',
                    default='strip')
parser.add_argument('--plugin-jardir',
                    help='Relative path under Unity project where GlimpsePlugin.jar will be copied (Android only, default = Assets/Plugins/Android)',
                    default='Assets/Plugins/Android')

parser.add_argument('unity_project',
                    help='Absolute path to a Unity project where we will install the Glimpse Unity plugin')
parser.add_argument('plugin_libdir',
                    help='Relative path under Unity project where libglimpse-unity-plugin.so and dependency libraries will be copied')

args = parser.parse_args()

introspect_cmd = os.environ['MESONINTROSPECT'].split()
targets_json = subprocess.check_output(introspect_cmd + [ '--targets', '.' ])

targets = json.loads(targets_json)
libs_blacklist = {
    'flann', # We only need libflann_cpp.so
    'glimpse_viewer_android'
}
if args.tango_libs:
    libs_blacklist.union({'fakenect'})

lib_strip_args = []
if args.buildtype == "release":
    # XXX: strip is failing with OSX builds currently so don't try and
    # strip for now...
    if sys.platform != 'darwin':
        lib_strip_args = [ '-s' ]
        if args.strip and args.strip != 'strip':
            lib_strip_args += [ '--strip-program', args.strip ]


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

            if parts[-1] == 'dylib':
                do_install(os.path.join(build_dir, target['filename']),
                           os.path.join(dst, parts[0][3:] + '.bundle'), lib_strip_args)
            else:
                do_install(os.path.join(build_dir, target['filename']),
                           os.path.join(dst, parts[0] + '.so'), lib_strip_args)

                if len(parts) > 2:
                    os.chdir(dst)
                    ln_cmd = [ 'ln', '-sf', parts[0] + '.so', parts[0] + '.so.' + parts[2] ]
                    print(" ".join(ln_cmd))
                    subprocess.check_call(ln_cmd)


def install_jar_target(jar_target_name, dst):
    for target in targets:
        if target['name'] == jar_target_name:
            basename = os.path.basename(target['filename'])
            do_install(os.path.join(build_dir, target['filename']),
                       os.path.join(dst, basename))


def install_unity_plugin__linux_so(dst, unity_project):
    install_shared_lib_targets(dst, libs_blacklist)

    # To avoid needing to set LD_LIBRARY_PATH for the plugin to load its
    # dependencies we update the RPATH to look in the same directory as
    # the plugin itself...
    os.chdir(dst)
    chrpath_cmd = [ 'chrpath', '-r', '$ORIGIN', 'libglimpse-unity-plugin.so' ]
    print(" ".join(chrpath_cmd))
    subprocess.check_call(chrpath_cmd)


def install_unity_plugin__android(dst, unity_project):
    install_shared_lib_targets(dst, libs_blacklist)

    unity_plugin_jar_dir = os.path.join(unity_project, args.plugin_jardir)
    os.makedirs(unity_plugin_jar_dir, exist_ok=True)
    install_jar_target('GlimpseUnity', unity_plugin_jar_dir)

    libcxx_shared_path = os.path.join(args.android_ndk,
            'sources/cxx-stl/llvm-libc++/libs',
            args.android_ndk_arch, 'libc++_shared.so')
    do_install(libcxx_shared_path, dst)

    if args.tango_libs:
        tango_libs = [ 'libtango_support_api.so' ]
        for lib in tango_libs:
            do_install(os.path.join(args.tango_libs, 'lib', args.android_ndk_arch, lib),
                       dst)


def install_unity_plugin__osx(dst, unity_project):
    install_shared_lib_targets(dst, libs_blacklist)


def install_unity_plugin_a(dst, target_name):
    for target in targets:
        if target['name'] == target_name:
            basename = os.path.basename(target['filename'])
            parts = basename.split('.')

            do_install(os.path.join(build_dir, target['filename']),
                       os.path.join(dst, parts[0] + '.a'), lib_strip_args)
            return


shared_plugin = True
for target in targets:
    if target['name'] == "glimpse-unity-plugin" and target['type'] == 'static library':
        shared_plugin = False
        break

unity_plugin_libs_dir = os.path.join(args.unity_project, args.plugin_libdir)
os.makedirs(unity_plugin_libs_dir, exist_ok=True)
if shared_plugin:
    if args.android_ndk_arch:
        install_unity_plugin__android(unity_plugin_libs_dir, args.unity_project)
    elif sys.platform == 'linux':
        install_unity_plugin__linux_so(unity_plugin_libs_dir, args.unity_project)
    elif sys.platform == 'darwin':
        install_unity_plugin__osx(unity_plugin_libs_dir, args.unity_project)
    else:
        sys.exit('Unhandled platform: %s' % sys.platform)
else:
    install_unity_plugin_a(unity_plugin_libs_dir, "glimpse-unity-plugin")
    install_unity_plugin_a(unity_plugin_libs_dir, "epoxy")
