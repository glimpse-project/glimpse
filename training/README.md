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

Training data can be contained in separate directories, but must have the same
names and directory structure inside those directories.

Building
========

The default make target will build all tools, but you likely want to build with
optimisations enabled:

CFLAGS="-O3 -mtune=native -march=native -g -Wall" make

Training a decision tree
========================

Run the tool 'train_rdt' to train a tree. Running it with no parameters, or
with the -h/--help parameter will print usage details, with details about the
default parameters.

For example, to train a decision tree using a shuffled set of 1000 training
images with 34 body-part labels, and generated from a camera with a vertical
FOV of 54.5 degrees:

train_rdt 54.5 34 path-to-label-imgs path-to-depth-imgs output.rdt -l 1000 -s

Creating a joint map
====================

To know what to do with the bones files, these tools need a joint-map file.
This is a human-readable plain text file that describes what bones map to which
labels.

When mapping bones to joints, it's possible to map the centre-point of the bone
by referring to the name without a '.head' or '.tail' suffix.

For example, suppose there are five bones in the JSON data, called 'head',
'left_arm', 'right_arm', 'left_leg' and 'right_leg'. Suppose we want to map
them to nine joints that correspond to the head, left and right shoulders,
left and right wrists, left and right hips and left and right ankles. We may
write a joint map like so (where labels have been picked arbitrarily):

head, 0, 1, 2, 3
left_arm.head, 4
left_arm.tail, 5
right_arm.head, 6
right_arm.tail, 7
left_leg.head, 8
left_leg.tail, 9
right_leg.head, 10
right_leg.tail, 11

Generating joint files
======================

Run the tool 'json-to-jnt.py' like so:

json-to-jnt.py joint-map.txt path-to-json-files

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
