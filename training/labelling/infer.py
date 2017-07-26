#!/usr/bin/env python3

import os
import cv2
import png
import sys
import pickle
import numpy as np

def get_depth(path):
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return exr[...,0]

def eval_pixel(node, depth, x):
    clip = np.array(np.shape(depth)) - 1
    dx = depth[x[0]][x[1]]

    while 't' in node:
        u = np.around(np.clip(np.add(x, np.divide(node['u'], dx)), [0, 0], clip)).astype(np.int32)
        v = np.around(np.clip(np.add(x, np.divide(node['v'], dx)), [0, 0], clip)).astype(np.int32)
        fx = depth[u[0]][u[1]] - depth[v[0]][v[1]]
        if fx < node['t']:
            node = node['l']
        else:
            node = node['r']

    return node['label_probs']

def eval_image(tree, depth):
    shape = np.shape(depth)
    rows, cols = np.indices(shape)
    coords = np.reshape(np.stack((rows, cols), axis=2), (shape[0] * shape[1], 2))

    output = {}
    n_pixels = len(coords)
    for x in coords:
        output[tuple(x)] = eval_pixel(tree, depth, x)
        n_pixels -= 1
        if n_pixels % 100 == 0:
            sys.stdout.write('%d pixels left %s     \r' % (n_pixels, str(tuple(x))))
            sys.stdout.flush()

    return output

def generate_png(label_probs, name):

    image = [[]]
    for coord, probs in label_probs.items():
        row = coord[0]
        col = coord[1]
        while row >= len(image):
            image.append([])
        while col >= len(image[row]):
            image[row].append(0)

        label = 0
        prob = -1
        for key, value in probs.items():
            if value > prob:
                prob = value
                label = key

        image[row][col] = int(label)

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
