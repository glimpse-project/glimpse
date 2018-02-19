#!/usr/bin/env python3

# Reference:
# https://www.apriorit.com/dev-blog/233-how-to-build-apk-file-from-command-line

import argparse
import json
import os
import stat
import subprocess
import sys
import glob
import shutil


if 'MESON_BUILD_ROOT' in os.environ:
    build_dir = os.environ['MESON_BUILD_ROOT']
    os.chdir(os.environ['MESON_BUILD_ROOT'])
else:
    build_dir = os.getcwd()

if 'MESON_SOURCE_ROOT' in os.environ:
    src_dir = os.environ['MESON_SOURCE_ROOT']
else:
    src_dir = os.path.abspath('..')

parser = argparse.ArgumentParser()
parser.add_argument('--buildtype',
                    help="The Meson configured buildtype",
                    default='debug')
parser.add_argument('--android-build-tools',
                    default='26.0.2',
                    help='The Android SDK Build Tools version to use (default is 26.0.2)')
parser.add_argument('--tango-libs',
                    help='Location of Tango libraries')
parser.add_argument('--android-ndk',
                    help='Path to the Android NDK',
                    default=os.getenv("ANDROID_NDK_HOME", ""))
parser.add_argument('--android-ndk-arch',
                    help='The lib/<arch>/ directory name to use in the .apk and used to find a copy of libc++_shared.so in the Android NDK (default armeabi-v7a)',
                    default='armeabi-v7a')
parser.add_argument('--android-sdk',
                    help='Path to the Android SDK',
                    default=os.getenv("ANDROID_HOME", ""))
parser.add_argument('--android-api',
                    help='Target Android API level (default is 23)',
                    type=int, default=23)
parser.add_argument('--strip',
                    help='The tool for stripping shared library symbols',
                    default='strip')
parser.add_argument('--keystore',
                    help='Keystore location (default ~/.android/debug.keystore)',
                    default=os.path.join(os.environ['HOME'], '.android', 'debug.keystore'))
parser.add_argument('--keystore_pass',
                    help='Keystore password (default "android")',
                    default='android')
parser.add_argument('--key_alias',
                    help='Keystore password (default "androiddebugkey")',
                    default='androiddebugkey')
parser.add_argument('manifest', help='Manifest file for package (relative paths are relative to the top of the repository)')
parser.add_argument('res', help='Path to res/ resources to include in APK (relative paths are relative to the top of the repository)')
parser.add_argument('name', help='Package Name')
parser.add_argument('jar_target', help='Jar target that should be packaged')
args = parser.parse_args()


build_tools_path = os.path.join(args.android_sdk, 'build-tools', args.android_build_tools)
android_jar_path = os.path.join(args.android_sdk, 'platforms',
        'android-%d' % args.android_api, 'android.jar')
aapt_path = os.path.join(build_tools_path, 'aapt')
zipalign_path = os.path.join(build_tools_path, 'zipalign')
dx_path = os.path.join(build_tools_path, 'dx')

if 'MESONINTROSPECT' in os.environ:
    introspect_cmd = os.environ['MESONINTROSPECT'].split()
else:
    introspect_cmd = [ 'meson', 'introspect' ]

targets_json = subprocess.check_output(introspect_cmd + [ '--targets' ])

targets = json.loads(targets_json)
libs_blacklist = {
    'flann', # We only need libflann_cpp.so
    'glimpse-unity-plugin'
}
#if args.tango_libs:
#    lib_blacklist.union({'fakenect'})

if args.buildtype == "release":
    lib_strip_args = [ '-s', '--strip-program', args.strip ]
else:
    lib_strip_args = []


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

            do_install(os.path.join(build_dir, target['filename']),
                       os.path.join(dst, parts[0] + '.so'), lib_strip_args)


def do_check_call(args):
    print("\nRunning: " + " ".join(args))
    subprocess.check_call(args)



# Stage files need for .apk...

stage_dir = args.name + '_apk_stage'
try:
    mode = os.stat(stage_dir).st_mode
    if stat.S_ISDIR(mode):
        shutil.rmtree(stage_dir)
except FileNotFoundError:
    pass

assets_path = os.path.join(stage_dir, 'assets')
bin_path = os.path.join(stage_dir, 'bin')
gen_path = os.path.join(stage_dir, 'gen')
lib_path = os.path.join(stage_dir, 'lib', args.android_ndk_arch)

os.mkdir(args.name + '_apk_stage')
os.mkdir(assets_path)
os.mkdir(bin_path)
os.mkdir(gen_path)
os.makedirs(lib_path)

install_shared_lib_targets(lib_path, libs_blacklist)

libcxx_shared_path = os.path.join(args.android_ndk,
        'sources/cxx-stl/llvm-libc++/libs',
        args.android_ndk_arch, 'libc++_shared.so')
do_install(libcxx_shared_path, lib_path)

if args.tango_libs:
    tango_libs = [ 'libtango_support_api.so' ]
    for lib in tango_libs:
        do_install(os.path.join(args.tango_libs, 'lib', args.android_ndk_arch, lib),
                lib_path)


# Create .apk...

unaligned_apk_name = args.name + '.unaligned.apk'
apk_name = args.name + '.apk'

manifest_path = os.path.join(src_dir, args.manifest)
res_path = os.path.join(src_dir, args.res)
os.chdir(stage_dir)

package_cmd = [
    aapt_path, 'package',
    '-v', # verbose
    '-f', # force overwrite of files
    '-I', android_jar_path,
    '-M', manifest_path,
    '-A', 'assets',
    '-S', res_path,
    '-m', '-J', 'gen',
    '-F', unaligned_apk_name]

if args.buildtype == "debug":
    package_cmd += [ '--debug-mode' ]

do_check_call(package_cmd)

do_check_call([
    'javac',
    '-classpath', android_jar_path,
    'gen/com/impossible/glimpse/' + args.name + '/R.java',
    '-d', 'bin'])

for target in targets:
    if target['name'] == args.jar_target:
        do_check_call([dx_path, '--dex', '--verbose', '--min-sdk-version=%d' % args.android_api, '--output=classes.dex', 'bin', os.path.join(build_dir, target['filename'])])

do_check_call([aapt_path, 'add', unaligned_apk_name, 'classes.dex' ])
libs_glob = glob.glob(os.path.join('lib', args.android_ndk_arch, '*'))
do_check_call([aapt_path, 'add', unaligned_apk_name ] + libs_glob)

# TODO: if api >= 23 then zipalign first and use apksigner
do_check_call(['jarsigner', '-verbose', '-sigalg', 'SHA1withRSA', '-digestalg', 'SHA1', '-keystore', args.keystore, '-storepass', args.keystore_pass, unaligned_apk_name, args.key_alias ])
do_check_call([zipalign_path, '-f', '4', unaligned_apk_name, apk_name ])
