#!/usr/bin/env python3

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

import os
import sys

try:
    import bpy
except:
    print("This script should be run via blender something like:")
    print("\n")
    print("  blender -b -P " + sys.argv[0] + " -- --help-script")
    print("\n")
    sys.exit(1)

import addon_utils
import textwrap
import argparse

if "--" in sys.argv:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser(prog="enable-addons", add_help=False,
        epilog=textwrap.dedent("""\
        A utility to help install the 'Glimpse Training Data Generator' Blender addon
        """))
parser.add_argument('--help-script', help='Show this help message and exit', action='help')
parser.add_argument('--info', help='Print various blender config paths and exit', action='store_true')
parser.add_argument('--set-scripts-dir', help='Set scripts directory in user preferences and exit', action='store_true')

args = parser.parse_args(argv)

prefs_scripts_dir = bpy.context.user_preferences.filepaths.script_directory
if args.info:
    print("BLENDER_USER_SCRIPTS=\"" + prefs_scripts_dir + "\"")
    print("BLENDER_USER_CONFIG=\"" + bpy.utils.script_path_user() + "\"")
    bpy.ops.wm.quit_blender()
    sys.exit(0)

if args.set_scripts_dir:
    glimpse_script_path = os.getcwd()

    if prefs_scripts_dir != "" and prefs_scripts_dir != glimpse_script_path:
        print("\n")
        print("Error:\n")
        print("Your Blender user preferences already set a script directory which")
        print("we don't want to trample:")
        print("\n")
        print("  dir = \"" + bpy.context.user_preferences.filepaths.script_directory + "\"");
        print("\n")
        print("An alternative approach to installing this addon is to create a symlink like:")
        print("\n")
        print("  $  mkdir -p " + os.path.join(bpy.utils.script_path_user(), "addons"))
        print("  $  cd " + os.path.join(bpy.utils.script_path_user(), "addons"))
        print("  $  ln -s " + os.path.join(os.getcwd(), "addons", "glimpse_data_generator"))
        print("\n")
        bpy.ops.wm.quit_blender()
        sys.exit(1)

    print("\n")
    print("Setting Blender user preferences, script directory = " + glimpse_script_path)
    print("\n")
    print("Enabling required addons in user preferences...\n")
    print("\n")
    bpy.context.user_preferences.filepaths.script_directory = glimpse_script_path
    bpy.ops.wm.save_userpref()
    sys.exit(0)

addon_dependencies = [
    'glimpse_data_generator',
    'mesh_paint_rig',
    'makeclothes',
    'maketarget',
    'makewalk',
]

def on_error(e):
    print("Failed to enable addon: " + str(e))
    sys.exit(1)

for addon in addon_dependencies:
    addon_utils.enable(addon, default_set=True, persistent=True, handle_error=on_error)

bpy.ops.wm.save_userpref()
