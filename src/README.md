Training a decision forest and joint inference parameters
=========================================================

Requirements
============

At this point it's assumed that you've used `glimpse-cli.py` to render some
training images. If not, please see the
[glimpse-training-data/README.md](https://github.com/glimpse-project/glimpse-training-data/blob/master/README.md)
for further details first.


Pre-process rendered images
===========================

Before starting training we process the images rendered by Blender so we can
increase the amount of training data we have by e.g. mirroring images and we
e.g add noise to make the data more represented of images captured by a
camera instead of being rendered.

If we have rendered data via glimpse-cli.py under
`/path/to/glimpse-training-data/render/generated/test-render` then these images
can be processed as follows:

```
./image-pre-processor \
    /path/to/glimpse-training-data/render/generated/test-render \
    /path/to/glimpse-training-data/render/pre-processed/test-render
```


Index frames to train with
==========================

For specifying which frames to train with, an index should be created with the
indexer.py script.

This script builds an index of all available rendered frames in a given
directory and can then split that into multiple sub sets with no overlap. For
example you could index three sets of 300k images out of a larger set of 1
million images for training three separate decision trees.

For example to create a single 'tree0' index of 100000 images you could run:
```
./indexer.py -i tree0 100000 /path/to/glimpse-training-data/render/pre-processed/test-render
```

This would create an `index.full` and `index.tree0` under the
`pre-processed/test-render/` directory.


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

By default the revision controlled `src/joint-map.json` file should be used

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
