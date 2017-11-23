Rendering training data
=======================

Requirements
============

* glimpse-training.blend
  - This .blend file is pre-loaded with lots of mocap data and a set
    of makehuman models and clothing.
* Glimpse/Make{human,clothes} addons set up
  - You must have followed the instructions in glimpse-sdk/blender/README.md
    to set up Blender so it knows where to find the Glimpse addon when it
    loads as well as the makehuman and makeclothes addons
* mocap data
  - We're using mocap data from Carnegie Mellon university converted to
    .bvh files which Blender understands and then indexed so we can
    potentially blacklist or tweak rendering parameters per-file

To render a new data set with blender we have a glimpse-cli.py script that
provides a basic command line interface to glimpse-training.blend

Unfortunately --help as an argument name clashes with another addon but
you can see an overview of the interface with --glimpse-help like:

```
blender -b \                            # run in the background
    /path/to/glimpse-training.blend \   # .blend file *before* .py script
    -P blender/glimpse-cmd.py \         # command line interface script
    -- \                                # remaining args for script
    --glimpse-help
```

The amount to render is measured in terms of indexed motion capture files
(re: index.json in the directory of mocap files)

To render mocap files from 0 to 100 you could run:

```
blender -b \
    /path/to/glimpse-training.blend \
    -P blender/glimpse-cmd.py \
    -- \
    --start 0 \
    --end 100 \
    --dest /path/to/render \
    --name "test-render" \
    /path/to/mocap/files/
```

Training a decision forest and joint inference parameters
=========================================================

Requirements
============

* Depth training images
  - Images in EXR format with a single, half-float 'Y' channel representing
    depth in meters.
* Body-part label images
  - 8-bit palettised or grayscale images with each pixel colour representing
    a particular body-part label.
* Bone position JSON files
  - See 'example-bones.json'
* A top-level training data meta.json
  - For determing the camera properties used to render the training data and
    looks like { 'camera': { 'width': 172, 'height':224, 'vertical_fov':60 }}
* An index.xyz file
  - For specifying which frames to train with, an index should be created
    with the training/indexer.py script.
    E.g. `./indexer.py -i tree0 100000 path/to/training-data`

Building
========

The default make target will build all tools, but you likely want to build with
optimisations enabled:

make RELEASE=1

Training a decision tree
========================

Run the tool 'train_rdt' to train a tree. Running it with no parameters, or
with the -h/--help parameter will print usage details, with details about the
default parameters.

For example, if you have an index.tree0 file at the top of your training data
you can train a decision tree like:

```
train_rdt path-training-data tree0 tree0.rdt
```

Creating a joint map
====================

To know which bones from the training data are of interest, and what body
labels they are associated with, these tools need a joint-map file.  This is a
human-readable JSON text file that describes what bones map to which labels.

It's an array of objects where each object specifies a joint and an array of
label indices. A joint name is comprised of a bone name follow by `.head` or
`.tail` to specify which end of the bone. For example:

```
[
    {
        "joint": "head.tail",
        "labels": [ 2, 3 ]
    },
    {
        "joint": "neck_01.head",
        "labels": [ 4 ]
    },
    {
        "joint": "upperarm_l.head",
        "labels": [ 7 ]
    },
    ...
]
```

By default the revision controlled `training/joint-map.json` file should be used

Generating joint files
======================

Run the tool 'json-to-jnt.py' like so:

json-to-jnt.py joint-map.json path-to-labels-directory

Binary files with the same names as the json files, but the extension '.jnt'
will be written to the same location as the json files. These files will be
used by the joint inference training program.

Training joint inference parameters
===================================

Run the tool 'train_joint_params' to train joint parameters. Running it with no
parameters, or with the -h/--help parameter will print usage details, with
details about the default parameters.

Note that this tool is currently much more CPU and memory intensive than the
decision tree training tool.

For example, to train joint parameters from a decision forest of two trees
named 'tree1.rdt' and 'tree2.rdt' using a shuffled set of 10 training images,
where the background label is '33':

train_joint_params path-to-depth-imgs path-to-jnt-files jointmap.txt \
                   output.jip -l 10 -s -g 33 -- tree1.rdt tree2.rdt

The joint parameter training program can also output the accuracy of the
decision forest when used for inferrence with the given depth images by
specifying a label directory that corresponds to the given depth image
directory. This can be accomplished with the -c/--label-dir parameter.
