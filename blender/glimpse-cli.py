# Run like:
# $ blender.exe -b /path/to/glimpse-training.blend -P generator-cmd.py -- -h

import sys
import argparse
import bpy

if "--" in sys.argv:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser(prog="glimpse-generator")

parser.add_argument('--info', help='Load the mocap index and print summary information', action='store_true')
parser.add_argument('--preload', help='Preload mocap files as actions before rendering', action='store_true')
parser.add_argument('--purge', help='Purge mocap actions', action='store_true')
parser.add_argument('--start', type=int, default=0, help='Index of first MoCap to render')
parser.add_argument('--end', default=0, type=int, help='Index of last MoCap to render')
args = parser.parse_args(argv)


bpy.ops.glimpse.open_bvh_index()

if args.start:
    bpy.context.scene.GlimpseBvhGenFrom = args.start

if args.end:
    bpy.context.scene.GlimpseBvhGenTo = args.end


if args.info:
    bpy.ops.glimpse.generator_info()
    bpy.ops.wm.quit_blender()

if args.preload:
    bpy.ops.glimpse.generator_preload()
    print("Saving to %s" % bpy.context.blend_data.filepath)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.context.blend_data.filepath)
    bpy.ops.wm.quit_blender()
    
if args.purge:
    bpy.ops.glimpse.purge_mocap_actions()
    basename = bpy.path.basename(bpy.context.blend_data.filepath)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath('//%s-purged.blend' % basename[:-6]))
    bpy.ops.wm.quit_blender()
    
import cProfile
cProfile.run("bpy.ops.glimpse.generate_data()", "c:\\tmp\\blender.prof")
 
import pstats
p = pstats.Stats("c:\\tmp\\blender.prof")
p.sort_stats("cumulative").print_stats(20)


print(str(bpy.ops.glimpse.generate_data))
print(bpy.context.scene.GlimpseBvhRoot)
