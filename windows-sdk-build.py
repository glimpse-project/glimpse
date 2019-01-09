#!/usr/bin/env python3

# Copyright (c) 2018 Robert Bragg <robert@sixbynine.org>

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
#

# This script builds a portable Windows SDK (headers and libraries) that
# can be used for cross compiling Windows applications from Linux.
#
# The script should be run under Microsoft' Windows Subsystem for Linux, with
# a Windows installation found under /mnt/c (or use --c-drive to change)
# assuming Visual Studio 2017 Community Edition and Windows 10 SDK have been
# installed.
#
# Note: the resulting SDK can be used from non-WSL-based Linux installations
# or potentially (untested) OSX.
#
# The output SDK must be written to a case-sensitive filesystem since it
# intentionally creates alternative header files that only differ in case to
# allow pre-existing internal SDK includes but also allow applications to use
# consistent lowercase include names.
#
# All included libraries have lowercase names
#
# The script will generate Meson toolchain configurations named:
#   windows-sdk/meson/windows-<arch>-debug-cross-file.txt
#   windows-sdk/meson/windows-<arch>-release-cross-file.txt
#
# By default these will refer to clang-8 (development branch at time of
# writing) but this can E.g. be changed with --llvm-version=7.
#
# Note The Meson toolchain configs include absolute paths so if the SDK is
# moved make sure to update these configs too. Re-running this script when
# there is a pre-existing output SDK directory will only update toolchain
# configs without modifying the SDK.
#
# See --help for details but in general just running ./windows-sdk-build.py
# with no arguments should hopefully Just Work to create and SDK under
# ./windows-sdk/
#

import os
import re
import sys
import stat
import argparse
import textwrap
from distutils import file_util
from distutils import dir_util
from distutils import log

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='windows-sdk',
                    help='Output directory')
parser.add_argument('--c-drive', default='/mnt/c',
                    help="Path to C drive where Windows is installed (default = '/mnt/c')")
default_vs_path = os.path.join('Program Files (x86)',
                               'Microsoft Visual Studio',
                               '2017')
parser.add_argument('--visual-studio-2017-path',
                    default=default_vs_path,
                    help="Path to Visual Studio 2017 installation (default = '%s')" %
                         default_vs_path)
parser.add_argument('--msvc-version',
                    help="Override version of MSVC headers/libs to copy (default = latest)")
default_sdk_path = os.path.join('Program Files (x86)', 'Windows Kits', '10')
parser.add_argument('--sdk-path',
                    default=default_sdk_path,
                    help="Path to windows SDK installation (default = '%s')" %
                         default_sdk_path)
parser.add_argument('--sdk-version',
                    help="Override version of SDK headers/libs to copy (default = latest)")
parser.add_argument('--arch', action='append', default=['x64'],
                    help="Control which library architectures to copy (default = x64)")

parser.add_argument('--llvm-version', default=8,
                    help="Override version of llvm version referenced in toolchain configurations (default = 8)")

parser.add_argument('--debug', action='store_true',
                    help='Show debug messages')

args = parser.parse_args()

if args.debug:
    log.set_verbosity(log.INFO)
    log.set_threshold(log.INFO)

out_dir = args.out

vc_include_dir = os.path.join(out_dir, 'vc', 'include')
vc_lib_dir = os.path.join(out_dir, 'vc', 'lib')
ucrt_include_dir = os.path.join(out_dir, 'ucrt', 'include')
ucrt_lib_dir = os.path.join(out_dir, 'ucrt', 'lib')
sdk_um_include_dir = os.path.join(out_dir, 'sdk', 'um', 'include')
sdk_shared_include_dir = os.path.join(out_dir, 'sdk', 'shared', 'include')
sdk_um_lib_dir = os.path.join(out_dir, 'sdk', 'um', 'lib')


if os.path.exists(out_dir):
    print("%s directory already exists - only updating toolchain configs..." % out_dir)
else:
    print("Writing Meson toolchain configs...")

meson_arch_cpu_family_map = {
    'x64': 'x86_64',
    'x86': 'x86'
}

# These are the alternative runtime libraries provided on Windows...
#
# Release DLLs   (/MD ): -lmsvcrt     -lvcruntime        -lucrt
# Debug DLLs     (/MDd): -lmsvcrtd    -lvcruntimed       -lucrtd
# Release Static (/MT ): libcmt.lib   libvcruntime.lib   libucrt.lib
# Debug Static   (/MTd): libcmtd.lib  libvcruntimed.lib  libucrtd.lib

template = """\
[binaries]
name = 'clang-{llvm_version}-windows-{arch}-{build_type}'
c = 'clang-{llvm_version}'
cpp = 'clang++-{llvm_version}'
ar = 'ar'
ld = 'ldd-{llvm_version}'
strip = 'strip'

[host_machine]
system = 'windows'
cpu_family = '{cpu_family}'
cpu = '{cpu}'
endian = 'little'

[properties]
sdk_path = '{sdk_path}'

c_args = [ '-target', 'x86_64-pc-windows', '-fms-extensions', '-fms-compatibility', '-fdelayed-template-parsing', '-Wno-expansion-to-defined', '-Wno-ignored-attributes', '-isystem', '{vc_include}', '-isystem', '{ucrt_include}', '-isystem', '{sdk_um_include}', '-isystem', '{sdk_shared_include}', '-D_WINSOCK_DEPRECATED_NO_WARNINGS=1', '-DNOMINMAX=1', '-D_CRT_DECLARE_NONSTDC_NAMES=1', '-D_CRT_NONSTDC_NO_DEPRECATE=1', '-D_CRT_SECURE_NO_DEPRECATE=1', '-D_CRT_NONSTDC_NO_WARNINGS=1', '-D_CRT_SECURE_NO_WARNINGS', '-fuse-ld=lld' ]

c_link_args = [ '-target', 'x86_64-pc-windows', '-fms-compatibility', '-L{vc_lib}', '-L{ucrt_lib}', '-L{sdk_um_lib}', '-fuse-ld=lld', '-nostdlib', {extra_link_args} ]

cpp_args = [ '-target', 'x86_64-pc-windows', '-fms-extensions', '-fms-compatibility', '-fdelayed-template-parsing', '-Wno-expansion-to-defined', '-Wno-ignored-attributes', '-isystem', '{vc_include}', '-isystem', '{ucrt_include}', '-isystem', '{sdk_um_include}', '-isystem', '{sdk_shared_include}', '-D_WINSOCK_DEPRECATED_NO_WARNINGS=1', '-DNOMINMAX=1', '-D_CRT_DECLARE_NONSTDC_NAMES=1', '-D_CRT_NONSTDC_NO_DEPRECATE=1', '-D_CRT_SECURE_NO_DEPRECATE=1', '-D_CRT_NONSTDC_NO_WARNINGS=1', '-D_CRT_SECURE_NO_WARNINGS=1', '-fuse-ld=lld' ]

cpp_link_args = [ '-target', 'x86_64-pc-windows', '-fms-compatibility', '-L{vc_lib}', '-L{ucrt_lib}', '-L{sdk_um_lib}', '-fuse-ld=lld', '-nostdlib', '-nostdlib++', {extra_link_args} ]
"""

os.makedirs(os.path.join(out_dir, 'meson'), exist_ok=True)
for arch in args.arch:
    for build_type in ['debug', 'release']:
        filename = 'windows-%s-%s-cross-file.txt' % (arch, build_type)
        with open(os.path.join(out_dir, 'meson', filename), 'w') as fp:
            if build_type == 'debug':
                link_args = "'-lmsvcrtd', '-lvcruntimed', '-lucrtd', '-g'"
            else:
                link_args = "'-lmsvcrt', '-lvcruntime', '-lucrt'"

            if arch in meson_arch_cpu_family_map:
                cpu_family = meson_arch_cpu_family_map[arch]
            else:
                cpu_family = arch
            cpu = cpu_family

            fp.write(template.format(
                build_type=build_type,
                llvm_version=str(args.llvm_version),
                arch=arch,
                cpu_family=cpu_family,
                cpu=cpu,
                sdk_path=os.path.abspath(out_dir),
                vc_include=os.path.abspath(vc_include_dir),
                ucrt_include=os.path.abspath(ucrt_include_dir),
                sdk_um_include=os.path.abspath(sdk_um_include_dir),
                sdk_shared_include=os.path.abspath(sdk_shared_include_dir),
                vc_lib=os.path.abspath(os.path.join(vc_lib_dir, arch)),
                ucrt_lib=os.path.abspath(os.path.join(ucrt_lib_dir, arch)),
                sdk_um_lib=os.path.abspath(os.path.join(sdk_um_lib_dir, arch)),
                extra_link_args=link_args
                ))
            print("> wrote meson/%s" % filename)

if os.path.exists(out_dir):
    sys.exit()

vc_dir = os.path.join(args.c_drive, args.visual_studio_2017_path)
if not os.path.exists(os.path.join(vc_dir, 'Community')):
    sys.exit("Visual Studio 2017 not found under %s" % vc_path)

msvc_path = os.path.join(vc_dir, 'Community', 'VC', 'Tools', 'MSVC')
msvc_ver = args.msvc_version

if not msvc_ver:
    for root, dirs, files in  os.walk(msvc_path):
        if len(dirs):
            dirs.sort()
            msvc_ver = dirs[-1]
        break

if not msvc_ver:
    sys.exit("Failed to find a version of MSVC under '%s'" % msvc_path)

print("Found MSVC version %s under '%s'" % (msvc_ver, msvc_path))

sdk_path = os.path.join(args.c_drive, args.sdk_path)
sdk_ver = args.sdk_version

if not sdk_ver:
    for root, dirs, files in  os.walk(os.path.join(sdk_path, 'Include')):
        versions = [name for name in dirs if name.startswith('10.')]
        if len(versions):
            versions.sort()
            sdk_ver = versions[-1]
        break

if not sdk_ver:
    sys.exit("Failed to find an SDK version under '%s'" % sdk_path)

print("Found SDK version %s under '%s'" % (sdk_ver, sdk_path))

print("Building SDK under '%s'" % out_dir)

os.makedirs(os.path.join(out_dir))
os.makedirs(vc_include_dir)
os.makedirs(ucrt_include_dir)
os.makedirs(sdk_um_include_dir)
os.makedirs(sdk_shared_include_dir)
if os.path.exists(os.path.join(out_dir, 'VC')):
    sys.exit("Can't build SDK under a case-insensitive filesystem")

copy_verbose=True if args.debug else False


print("Copying MSVC %s headers..." % msvc_ver)
dir_util.copy_tree(os.path.join(msvc_path, msvc_ver, 'include'),
                   os.path.join(out_dir, 'vc', 'include'),
                   verbose=copy_verbose)
print("Copying Universal C Runtime %s headers..." % sdk_ver)
dir_util.copy_tree(os.path.join(sdk_path, 'Include', sdk_ver, 'ucrt'),
                   os.path.join(out_dir, 'ucrt', 'include'),
                   verbose=copy_verbose)
print("Copying User Mode SDK %s headers..." % sdk_ver)
dir_util.copy_tree(os.path.join(sdk_path, 'Include', sdk_ver, 'um'),
                   os.path.join(out_dir, 'sdk', 'um', 'include'),
                   verbose=copy_verbose)
print("Copying Shared (User/Kernel Mode) SDK %s headers..." % sdk_ver)
dir_util.copy_tree(os.path.join(sdk_path, 'Include', sdk_ver, 'shared'),
                   os.path.join(out_dir, 'sdk', 'shared', 'include'),
                   verbose=copy_verbose)

# Unlike for headers we can't afford to have multiple copies with different
# names so we force all libraries to have a lowercase name...
def copy_libs(src, dest):
    os.makedirs(dest)
    for root, dirs, files in os.walk(src):
        if root != src:
            continue
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith('.lib') or lower_name.endswith('.pdb'):
                if args.debug:
                    print("> %s -> %s/%s" % (name, dest, lower_name))
                file_util.copy_file(os.path.join(root, name), os.path.join(dest, lower_name))


for arch in args.arch:
    print("Copying MSVC %s %s libs..." % (msvc_ver, arch))
    copy_libs(os.path.join(msvc_path, msvc_ver, 'lib', arch),
              os.path.join(out_dir, 'vc', 'lib', arch))
    print("Copying Universal C Runtime %s %s libs..." % (sdk_ver, arch))
    copy_libs(os.path.join(sdk_path, 'Lib', sdk_ver, 'ucrt', arch),
              os.path.join(out_dir, 'ucrt', 'lib', arch))
    print("Copying User Mode SDK %s %s libs..." % (sdk_ver, arch))
    copy_libs(os.path.join(sdk_path, 'Lib', sdk_ver, 'um', arch),
              os.path.join(out_dir, 'sdk', 'um', 'lib', arch))


print("Removing MSVC intrinsics headers (they conflict with Clang's own headers)...")
for root, dirs, files in os.walk(os.path.join(out_dir, 'vc', 'include')):
    for name in files:
        if 'intrin' in name:
            intrin_header = os.path.join(root, name)
            if args.debug:
                print("Removing %s" % intrin_header)
            os.remove(intrin_header)

    # The intrin0.h header that's part of the VC toolchain contains a minimal
    # subset of intrinsics but since it's not provided by Clang we simply make
    # it an alias for <intrin.h>...
    with open(os.path.join(out_dir, 'vc', 'include', 'intrin0.h'), 'w') as fp:
        if args.debug:
            print("Creating intrin0.h expected by some Microsoft headers, but not provided by Clang")

        fp.write(textwrap.dedent('''\
            #pragma once
            #include <intrin.h>
            '''))

include_dirs = [
    os.path.join(out_dir, 'vc', 'include'),
    os.path.join(out_dir, 'ucrt', 'include'),
    os.path.join(out_dir, 'sdk', 'um', 'include'),
    os.path.join(out_dir, 'sdk', 'shared', 'include'),
]


# If building on Linux with a case sensitive filesystem then we hit lots of
# problems due to include directive header names not matching the case of the
# filesystem.
#
# For a lot of cases it's simple enough to ensure there is a lowercase copy of
# each header.
#
# For other cases we actually parse all the headers and try and pluck the names
# used for all include directives and build a list of all the case variations we
# see for a particular normalized/lowercase name. We can then make sure all these
# variations exist too.

camel_names = {}

print("Indexing CamelCase include names...")
for include_dir in include_dirs:
    for root, dirs, files in os.walk(include_dir):
        for name in files:
            lower_name = name.lower()

            if not (lower_name.endswith('.h') or lower_name.endswith('.hxx')):
                continue

            if args.debug:
                print("Parsing %s" % os.path.join(root, name))
            with open(os.path.join(root, name)) as fp:
                lines = fp.readlines()
                for line in lines:
                    stripped = line.strip()
                    if not stripped.startswith('#'):
                        continue
                    packed = ''.join(stripped.split())
                    if not packed.startswith('#include'):
                        continue
                    tokens = [tok for tok in re.split('<|>|"', packed) if len(tok)]
                    if len(tokens) >= 2:
                        lower_inc = tokens[1].lower()
                        if tokens[1] == lower_inc:
                            continue
                        if lower_inc not in camel_names:
                            camel_names[lower_inc] = []
                        camel_names[lower_inc].append(tokens[1])


print("Adding lowercase and alternative CamelCase header files...")
for include_dir in include_dirs:
    for root, dirs, files in os.walk(include_dir):
        for name in files:
            lower_name = name.lower()

            if not (lower_name.endswith('.h') or lower_name.endswith('.hxx')):
                continue

            if lower_name != name:
                if args.debug:
                    print("Adding lowercase alternative for %s" % os.path.join(root, name))
                file_util.copy_file(os.path.join(root, name), os.path.join(root, lower_name))
            if lower_name in camel_names:
                for alt_name in camel_names[lower_name]:
                    if alt_name == lower_name or alt_name == name:
                        continue
                    if args.debug:
                        print("Adding %s alternative for %s" % (alt_name, os.path.join(root, name)))
                    file_util.copy_file(os.path.join(root, name), os.path.join(root, alt_name))


os.rename(os.path.join(out_dir, 'sdk', 'um', 'include', 'gl'),
          os.path.join(out_dir, 'sdk', 'um', 'include', 'GL'))


