#!/usr/bin/env python3

import glimpse
import sys

if len(sys.argv) != 5:
    print('Usage: infer.py <jointmap.txt> <params.jip> <depth.exr> <tree1.rdt> [tree2.rdt ...]')
    sys.exit(0)

forest = glimpse.Forest(sys.argv[4:])
jointmap = glimpse.JointMap(sys.argv[1], sys.argv[2])
depth_image = glimpse.DepthImage(sys.argv[3])

joints = jointmap.inferJoints(forest, depth_image)

for i in range(len(joints)):
    print('Joint %d: %f, %f, %f' % (i, joints[i][0], joints[i][1], joints[i][2]))
