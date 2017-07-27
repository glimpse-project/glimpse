#!/usr/bin/env python3

import os
import cv2
import png
import sys
import pickle
import numpy as np
from collections import deque

def get_depth(path):
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return exr[...,0]

def eval_image(node, depth):
    depth = np.array(depth)
    shape = np.array(np.shape(depth))
    clip = shape - 1

    n_pixels = shape[0] * shape[1]
    coords = deque(np.reshape(np.stack(np.indices(shape), axis=2), [n_pixels, 2]))

    output = [[None for x in range(shape[1])] for y in range(shape[0])]
    nodes = deque([node for i in range(n_pixels)])

    unrolled_depth = np.reshape(depth, [n_pixels])
    pixels = deque(np.stack([unrolled_depth, unrolled_depth], axis=1))

    current_depth = 0
    while len(nodes) > 0:
        nodes_u = [node['u'] for node in nodes]
        nodes_v = [node['v'] for node in nodes]

        u = np.around(np.clip(np.add(coords, np.divide(nodes_u, pixels)), [0, 0], clip)).astype(np.int32)
        v = np.around(np.clip(np.add(coords, np.divide(nodes_v, pixels)), [0, 0], clip)).astype(np.int32)

        nodes = deque( \
            [node['l'] if depth[u[0]][u[1]] - depth[v[0]][v[1]] < node['t'] else \
             node['r'] for node, u, v in zip(nodes, u, v)])

        # Check if we've hit any leaves
        for i in range(len(nodes)):
            node = nodes.popleft()
            coord = coords.popleft()
            pixel = pixels.popleft()

            if 't' not in node:
                output[coord[0]][coord[1]] = node['label_probs']
                n_pixels -= 1
            else:
                nodes.append(node)
                coords.append(coord)
                pixels.append(pixel)

        sys.stdout.write('(%d) %d pixels left       \r' % (current_depth, n_pixels))
        sys.stdout.flush()
        current_depth += 1

    return output

def generate_png(label_probs, name):
    image = [[0 for x in range(len(label_probs[0]))] for y in range(len(label_probs))]
    for y in range(len(label_probs)):
        for x in range(len(label_probs[y])):

            probs = label_probs[y][x]
            label = 0
            prob = -1
            for key, value in probs.items():
                if value > prob:
                    prob = value
                    label = key

            image[y][x] = int(label)

    with open(name, 'wb') as f:
        w = png.Writer(len(image[0]), len(image), greyscale=True)
        w.write(f, image)

# Open decision tree
with open(sys.argv[1], 'rb') as f:
    tree = pickle.load(f, encoding='latin1')

# Open depth image
depth = get_depth(sys.argv[2])

# Evaluate each pixel in the image
print('Evaluating image...')
label_probs = eval_image(tree, depth)

# Write that out and generate a png showing the most likely result
print('Writing out probability map')
with open('output.pkl', 'wb') as f:
    pickle.dump(label_probs, f)

print('Writing out png of likeliest labels')
generate_png(label_probs, 'output.png')
