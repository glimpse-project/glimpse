#!/usr/bin/env python

import os
import png
import sys
import Imath
import pickle
import numpy as np
import OpenEXR as exr

def get_depth(path):
    exrfile = exr.InputFile(path)
    hdr = exrfile.header()
    binary = exrfile.channel(hdr['channels'].keys()[0], Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.fromstring(binary, dtype=np.float32)
    depth = np.reshape(depth, (hdr['dataWindow'].max.y + 1, hdr['dataWindow'].max.x + 1))
    exrfile.close()
    return depth

def eval_pixel(tree, depth, x, node_path = ''):
    node = tree[node_path]
    if 't' in node:
        clip = np.array(np.shape(depth)) - 1
        dx = depth[x[0]][x[1]]
        u = np.clip(np.add(x, np.divide(node['u'], dx)), [0, 0], clip).astype(np.int32)
        v = np.clip(np.add(x, np.divide(node['v'], dx)), [0, 0], clip).astype(np.int32)
        fx = depth[u[0]][u[1]] - depth[v[0]][v[1]]
        if fx < node['t']:
            return eval_pixel(tree, depth, x, node_path + 'l')
        else:
            return eval_pixel(tree, depth, x, node_path + 'r')
    else:
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
        while (col + 1) * 3 > len(image[row]):
            image[row].append(0)

        label = 0
        prob = -1
        for key, value in probs.items():
            if value > prob:
                prob = value
                label = key

        label = int(label)
        r = label >> 16
        g = (label >> 8) & 0xFF
        b = label & 0xFF

        image[row][(col * 3)] = r
        image[row][(col * 3) + 1] = g
        image[row][(col * 3) + 2] = b

    with open(name, 'w') as f:
        w = png.Writer(len(image[0]) / 3, len(image))
        w.write(f, image)

# Open decision tree
with open(sys.argv[1], 'r') as f:
    tree = pickle.load(f)

# Open depth image
depth = get_depth(sys.argv[2])

# Evaluate each pixel in the image
print 'Evaluating image...'
label_probs = eval_image(tree, depth)

# Write that out and generate a png showing the most likely result
print 'Writing out probability map'
with open('output.pkl', 'w') as f:
    pickle.dump(label_probs, f)

print 'Writing out png of likeliest labels'
generate_png(label_probs, 'output.png')
