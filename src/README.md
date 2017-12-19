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
  - For determining the camera properties used to render the training data and
    looks like { 'camera': { 'width': 172, 'height':224, 'vertical_fov':60 }}
* An index.xyz file
  - For specifying which frames to train with, an index should be created
    with the indexer.py script.
    E.g. `./indexer.py -i tree0 100000 path/to/rendered-training-data`


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

It's an array of objects where each object specifies a joint, an array of
label indices and an array of other joints it connects to. A joint name is
comprised of a bone name follow by `.head` or `.tail` to specify which end of
the bone. For example:

```
[
    {
        "joint": "head.tail",
        "labels": [ 2, 3 ],
        "connections": [ "neck_01.head" ]
    },
    {
        "joint": "neck_01.head",
        "labels": [ 4 ],
        "connections": [ "upperarm_l.head" ]
    },
    {
        "joint": "upperarm_l.head",
        "labels": [ 7 ],
        "connections": []
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
