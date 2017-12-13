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

# Run like:
# $ blender.exe -b /path/to/glimpse-training.blend -P glimpse-cli.py -- --help-glimpse



import os
import sys
import argparse
import subprocess

try:
    import bpy
    import addon_utils
    as_blender_addon = True
except:
    as_blender_addon = False

if as_blender_addon:
    add_help_glimpse_opt = False
else:
    add_help_opt = True

if as_blender_addon:
    parser = argparse.ArgumentParser(prog="glimpse-generator", add_help=False)
    parser.add_argument('--help-glimpse', help='Show this help message and exit', action='help')
else:
    parser = argparse.ArgumentParser(prog="glimpse-generator")

parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--info', help='Load the mocap index and print summary information', action='store_true')
parser.add_argument('--preload', help='Preload mocap files as actions before rendering', action='store_true')
parser.add_argument('--purge', help='Purge mocap actions', action='store_true')
parser.add_argument('--start', type=int, default=0, help='Index of first MoCap to render')
parser.add_argument('--end', default=0, type=int, help='Index of last MoCap to render')
parser.add_argument('--dest', default=os.getcwd(), help='Directory to write files too')
parser.add_argument('--name', default=os.getcwd(), help='Unique name for this render run')
parser.add_argument('training_data', help='Directory with all training data')



#def run_cmd(args):
#    if cli_args.debug:
#        print("# " + " ".join(map(str, args)), file=sys.stderr)
#    try:
#        output = subprocess.check_output(args).decode("utf-8").strip()
#    except:
#        output = ""
#    if cli_args.debug:
#        print("# > " + "\n# > ".join(output.splitlines()))
#
#    return output


def run_cmd(args):
    if cli_args.debug:
        print("# " + " ".join(map(str, args)), file=sys.stderr)
        returncode = subprocess.call(args)
        print("# return status = " + str(returncode))
        return returncode
    else:
        return subprocess.call(args)

if as_blender_addon:
    if "--" in sys.argv:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    cli_args = parser.parse_args(argv)
else:
    cli_args = parser.parse_args()
    ret = run_cmd(['blender', '-b',
                   os.path.join(cli_args.training_data, 'glimpse-training.blend'),
                   '-P',
                   sys.argv[0],
                   '--'] +
                   sys.argv[1:])
    sys.exit(ret)


addon_dependencies = [
    'glimpse_data_generator',
    'mesh_paint_rig',
    'makeclothes',
    'maketarget',
    'makewalk',
]

dep_error = ""
for dep in addon_dependencies:
    addon_status = addon_utils.check(dep)
    if addon_status[0] != True or addon_status[1] != True:
        dep_error += "Addon '" + dep + "' has not been enabled through Blender's User Preferences\n"

if dep_error != "":
    print("\n")
    print("Error:\n")
    print(dep_error)
    print("\n")
    print("Please find the instructions for setting up the required addons in blender/README.md")
    print("> https://github.com/glimpse-project/glimpse/blob/master/blender/README.md")
    print("\n")
    sys.exit(1)

mocaps_dir = os.path.join(cli_args.training_data, 'mocap')
if not os.path.isdir(mocaps_dir):
    print("Non-existent mocaps directory %s" % mocaps_dir)
    bpy.ops.wm.quit_blender()
    sys.exit(1)
bpy.context.scene.GlimpseBvhRoot = mocaps_dir

bpy.ops.glimpse.open_bvh_index()

if cli_args.start:
    bpy.context.scene.GlimpseBvhGenFrom = cli_args.start

if cli_args.end:
    bpy.context.scene.GlimpseBvhGenTo = cli_args.end

if cli_args.info:
    bpy.ops.glimpse.generator_info()
    bpy.ops.wm.quit_blender()

if cli_args.preload:
    bpy.ops.glimpse.generator_preload()
    print("Saving to %s" % bpy.context.blend_data.filepath)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.context.blend_data.filepath)
    bpy.ops.wm.quit_blender()
    
if cli_args.purge:
    bpy.ops.glimpse.purge_mocap_actions()
    bpy.ops.wm.save_as_mainfile(filepath=bpy.context.blend_data.filepath)
    bpy.ops.wm.quit_blender()

if cli_args.dest == "":
    print("--dest argument required in this case to find files to preload")
    bpy.ops.wm.quit_blender()
if not os.path.isdir(cli_args.dest):
    print("Non-existent dest directory %s" % cli_args.dest)
    bpy.ops.wm.quit_blender()
bpy.context.scene.GlimpseDataRoot = cli_args.dest

print("DataRoot: " + cli_args.dest)

if cli_args.name == "":
    print("--name argument required in this case to find files to preload")
    bpy.ops.wm.quit_blender()
bpy.context.scene.GlimpseGenDir = cli_args.name


import cProfile
cProfile.run("bpy.ops.glimpse.generate_data()", "glimpse-" + cli_args.name + ".prof")
 
import pstats
p = pstats.Stats("glimpse-" + cli_args.name + ".prof")
p.sort_stats("cumulative").print_stats(20)


print(str(bpy.ops.glimpse.generate_data))
print(bpy.context.scene.GlimpseBvhRoot)
