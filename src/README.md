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
e.g add noise to make the data more representative of images captured by a
camera instead of being rendered.

If we have rendered data via `glimpse-cli.py` under
`/path/to/glimpse-training-data/renders/test-render` then these images
can be processed as follows:

```
image-pre-processor \
    /path/to/glimpse-training-data/renders/test-render \
    /path/to/glimpse-training-data/pre-processed/test-render \
    /path/to/glimpse-training-data/label-maps/2018-06-render-to-2018-06-rdt-map.json
```

The `2018-06-render-to-2018-06-rdt-map.json` argument defines a mapping from
the label values found within .png images under the `labels/` subdirectory of
the rendered training data and the packed/sequential indices that will be used
while training.

The file follows a schema like:
```
[
    {
        "name": "background",
        "inputs": [ 64 ]
    },
    {
        "name": "head left",
        "inputs": [ 7 ],
        "opposite": "head right"
    },
    {
        "name": "head right",
        "inputs": [ 15 ],
        "opposite": "head left"
    },
    {
        "name": "head top left",
        "inputs": [ 22 ],
        "opposite": "head top right"
    },
    {
        "name": "head top right",
        "inputs": [ 29 ],
        "opposite": "head top left"
    },
    {
        "name": "neck",
        "inputs": [ 36 ]
    },
    ...
]
```

*Note: the "opposite" property allows the image-pre-processor to automatically
flip images horizonally*

Normally `glimpse-training-data/label-maps/2018-06-render-to-2018-06-rdt-map.json`
can be used.


Index frames to train with
==========================

For specifying which frames to train with, an index should be created with the
`src/indexer.py` script.

This script builds an index of all available rendered frames in a given
directory and can then split that into multiple sub sets with no overlap. For
example you could index three sets of 300k images out of a larger set of 1
million images for training three separate decision trees.

For example to create a 'test' index of 10000 images you could run:
```
indexer.py -i test 10000 /path/to/glimpse-training-data/pre-processed/test-render
```
(*Note: this will also automatically create an `index.full` file*)

and then create three tree index files (sampled with replacement, but excluding
the test set images):
```
indexer.py \
    -e test \
    -i tree0 100000 \
    -i tree1 100000 \
    -i tree2 100000 \
    /path/to/glimpse-training-data/pre-processed/test-render
```
*Note: there may be overlapping frames listed in tree0, tree1 and tree1 but
none of them will contain test-set frames. See --help for details.*

Training a decision tree
========================

Run the tool `train_rdt` to train a tree. Running it with no parameters, or
with the `-h/--help` parameter will print usage details, with details about the
default parameters.

For example, if you have an index.tree0 file at the top of your training data
you can train a decision tree like:

```
train_rdt path-training-data tree0 tree0.json
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

Training joint inference parameters
===================================

Run the tool `train_joint_params` to train joint parameters. Running it with no
parameters, or with the `-h/--help` parameter will print usage details, with
details about the default parameters.

Note that this tool doesn't currently scale to handling as many images as
the decision tree training tool so it's recommended to create a smaller
dedicated index for training joint params.

For example, if you have an `index.joint-param-training` file then to train
joint parameters from a decision forest of two trees named `tree0.json` and
`tree1.json` you could run:

```
train_joint_params /path/to/glimpse-training-data/pre-processed/test-render \
                   joint-param-training \
                   src/joint-map.json \
                   output.jip -- tree0.json tree1.json
```


Convert .json trees to .rdt for runtime usage
=============================================

To allow faster loading of decision trees at runtime we have a simple binary
`.rdt` file format for trees.

For example, to create an `tree0.rdt` file from a `tree0.json` you can run:
```
json-to-rdt tree0.json tree0.rdt
```

*Note: `.rdt` files only include the information needed at runtime and so
training tools don't support loading these files.*

*Note: We don't aim to support forwards compatibility for `.rdt` besides having
a version check that lets us recognise incompatibility. Newer versions of
Glimpse may require you to recreate `.rdt` files.
