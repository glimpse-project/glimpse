#!/usr/bin/env python

# Points for optimisation
# - [DONE] Asynchronous image decoding in queue, rather than in graph
# - Remove, or reduce usage of feed_dict
# - Pre-process images before starting (decode, decolorize, resize?)
# - Carry over shannon entropy calculations from parent node
# - Process multiple images (or nodes?) at once (NB: Use of unique_with_count
#   may make this impossible with the current method)

import os
import Imath
import pickle
import StringIO
import numpy as np
import OpenEXR as exr
import tensorflow as tf

# Image dimensions
WIDTH=1080
HEIGHT=1920
# Number of images to pre-load in the queue
QUEUE_BUFFER=50
# Number of threads to use when pre-loading queue
QUEUE_THREADS=1
# Number of epochs to train per node
N_EPOCHS=100
# Depth to train tree to
MAX_DEPTH=20
# Maximum probe offset for random u/v values, in pixel meters
MAX_UV = 129
# Maximum value to set initial random t value at
MAX_T = 0.5
# Number of pixel samples
N_SAMP = 2000

def find_files(base_dir, extensions, name=None):
    for root, dirs, files in os.walk(base_dir):
        for basename in files:
            if name and not basename.startswith(name):
                continue
            for ext in extensions:
                if basename.endswith('.' + ext):
                    filename = os.path.join(root, basename)
                    yield filename
                    break

def read_depth(data):
    exrfile = exr.InputFile(StringIO.StringIO(data))
    hdr = exrfile.header()
    binary = exrfile.channel(hdr['channels'].keys()[0], Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.fromstring(binary, dtype=np.float32)
    depth = np.reshape(depth, (hdr['dataWindow'].max.y + 1, hdr['dataWindow'].max.x + 1, 1))
    exrfile.close()
    return depth

def get_offset_indices(image, pixels, x, u, v):
    extents = tf.subtract(tf.shape(image)[0:2], 1)
    n_pixels = tf.shape(pixels)[0]
    clip = tf.reshape(tf.tile(extents, [n_pixels]), [n_pixels, 2])
    pixels = tf.cast(tf.reshape(pixels, [n_pixels]), tf.float32)

    ux = tf.divide(tf.tile([u[0]], [n_pixels]), pixels)
    uy = tf.divide(tf.tile([u[1]], [n_pixels]), pixels)
    u = tf.stack((ux, uy), axis=1)

    vx = tf.divide(tf.tile([v[0]], [n_pixels]), pixels)
    vy = tf.divide(tf.tile([v[1]], [n_pixels]), pixels)
    v = tf.stack((vx, vy), axis=1)

    x = tf.cast(x, tf.float32)
    uindices = tf.clip_by_value(tf.cast(tf.add(x, u), tf.int32), 0, clip)
    vindices = tf.clip_by_value(tf.cast(tf.add(x, v), tf.int32), 0, clip)

    return uindices, vindices

def decolorize(pixels):
    r, g, b = tf.unstack(tf.cast(pixels, tf.int32), axis=1)
    return ((r * 256 * 256) + (g * 256) + b)

def shannon_entropy(values):
    y, idx, count = tf.unique_with_counts(values)
    ncount = tf.cast(count, tf.float32) / tf.cast(tf.reduce_sum(count), tf.float32)
    return -tf.reduce_sum(ncount * (tf.log(ncount) / tf.log(2.0))), y, ncount

# Collect training data
label_images = []
depth_images = []
for imagefile in find_files('training/color', ['png']):
    label_images.append(imagefile)
for imagefile in find_files('training/depth', ['exr']):
    depth_images.append(imagefile)

n_images = len(label_images)
assert(n_images == len(depth_images))

# Setup file readers
label_files = tf.train.string_input_producer(label_images, shuffle=False)
depth_files = tf.train.string_input_producer(depth_images, shuffle=False)
label_reader = tf.WholeFileReader()
depth_reader = tf.WholeFileReader()

# Setup image loaders
depth_key, depth_value = depth_reader.read(depth_files)
label_key, label_value = label_reader.read(label_files)

depth_image = tf.py_func(read_depth, [depth_value], tf.float32, stateful=False)
label_image = tf.image.decode_png(label_value, channels=3)

image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, \
                           dtypes=(tf.float32, tf.uint8))
enqueue = image_queue.enqueue((depth_image, label_image))

qr = tf.train.QueueRunner(image_queue, [enqueue] * QUEUE_THREADS)
tf.train.add_queue_runner(qr)

# Read in depth and label image
depth_image, label_image = image_queue.dequeue()
depth_image.set_shape([HEIGHT, WIDTH, 1])
label_image.set_shape([HEIGHT, WIDTH, 3])

# Pick splitting candidates (1)
u = tf.placeholder(tf.float32, shape=[2], name='u')
v = tf.placeholder(tf.float32, shape=[2], name='v')
t = tf.placeholder(tf.float32, shape=[1], name='t')

# Pixel coordinates to test
x = tf.placeholder(tf.int32, shape=[None, 2], name='x')

# Partition pixels into two groups (2)
depth_pixels = tf.gather_nd(depth_image, x)
uindices, vindices = get_offset_indices(depth_image, depth_pixels, x, u, v)
fdepth_pixels = tf.gather_nd(depth_image, uindices) - tf.gather_nd(depth_image, vindices)
# TODO: Find out if it's possible to do this comparison only once
lindex, _ = tf.split(tf.where(tf.less(fdepth_pixels, t)), 2, axis=1)
rindex, _ = tf.split(tf.where(tf.greater_equal(fdepth_pixels, t)), 2, axis=1)

# Compute gain (3)
label_pixels = decolorize(tf.gather_nd(label_image, x))
hq, x_labels, x_label_prob = shannon_entropy(label_pixels)
q = tf.cast(tf.shape(label_pixels)[0], tf.float32)

llabel_pixels = tf.gather_nd(label_pixels, lindex)
hql, l_labels, l_label_prob = shannon_entropy(llabel_pixels)
ql = tf.cast(tf.shape(llabel_pixels)[0], tf.float32)

rlabel_pixels = tf.gather_nd(label_pixels, rindex)
hqr, r_labels, r_label_prob = shannon_entropy(rlabel_pixels)
qr = tf.cast(tf.shape(rlabel_pixels)[0], tf.float32)

G = hq - ((ql / q * hql) + (qr / q * hqr))

# Initialise and run the session
init = tf.global_variables_initializer()
session = tf.Session()

with session.as_default():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    session.run(init)

    initial_coords = np.stack((np.random.random_integers(0, HEIGHT-1, N_SAMP * n_images), \
                               np.random.random_integers(0, WIDTH-1, N_SAMP * n_images)), \
                              axis=1)
    queue = [{ 'name': '', \
               'x': np.reshape(initial_coords, (n_images, 2000, 2)) }]
    tree = {}

    while len(queue) > 0:
        current = queue.pop(0)
        best_G = None
        best_u = None
        best_v = None
        best_t = None
        best_label_probs = None
        best_isNotLeaf = False
        nextNodes = []
        for epoch in range(N_EPOCHS):
            print 'Running epoch %d on node \'%s\'...' % (epoch, current['name'])

            # Initialise trial splitting candidates
            c_u = [np.random.randint(-MAX_UV, MAX_UV), \
                   np.random.randint(-MAX_UV, MAX_UV)]
            c_v = [np.random.randint(-MAX_UV, MAX_UV), \
                   np.random.randint(-MAX_UV, MAX_UV)]
            c_t = [np.random.random() * MAX_T]

            # Keep track of left/right modified coordinates for next node
            lcoords = []
            rcoords = []

            eG = 0
            eIsNotLeaf = False
            eXLabelProbs = []
            for i in range(n_images):
                # Bail out if we have no pixels to check
                if len(current['x'][i]) < 2:
                    lcoords.append([])
                    rcoords.append([])
                    continue

                t_G, t_l, t_r, t_labels, t_label_probs = \
                    session.run((G, lindex, rindex, x_labels, x_label_prob), \
                                feed_dict={u: c_u, v: c_v, t: c_t, \
                                           x: current['x'][i]})
                eG += t_G
                lcoords.append(current['x'][i][np.reshape(t_l, np.shape(t_l)[0])])
                rcoords.append(current['x'][i][np.reshape(t_r, np.shape(t_r)[0])])
                eXLabelProbs.append(np.stack((t_labels, t_label_probs), axis=1))
                if len(lcoords[-1]) > 1 and len(rcoords[-1]) > 1:
                    eIsNotLeaf = True

            eG /= n_images

            if best_G is None or eG > best_G:
                best_G = eG
                best_u = c_u
                best_v = c_v
                best_t = c_t
                best_isNotLeaf = eIsNotLeaf
                best_label_probs = eXLabelProbs
                nextNodes = [{ 'name': current['name'] + 'l', \
                               'x': lcoords }, \
                             { 'name': current['name'] + 'r',
                               'x': rcoords }]
                print '\tG = ' + str(best_G)
                print '\tt = ' + str(best_t)
                print '\tu = ' + str(best_u)
                print '\tv = ' + str(best_v)

        # Add the next nodes to train to the queue
        # TODO: Is the logic here correct?
        if best_isNotLeaf and \
           len(current['name']) < MAX_DEPTH:
            current['u'] = best_u
            current['v'] = best_v
            current['t'] = best_t
            queue.extend(nextNodes)
        else:
            # This is a leaf node, store the probabilities
            print 'Leaf node:'
            label_probs = {}
            for probs in best_label_probs:
                for prob in probs:
                    key = prob[0]
                    if key not in label_probs:
                        label_probs[key] = prob[1]
                    else:
                        label_probs[key] += prob[1]
            for key, value in label_probs.items():
                label_probs[key] = value / len(best_label_probs)
                print '%8d - %0.3f' % (key, label_probs[key])

            current['label_probs'] = label_probs

        # Save finished node to tree
        tree[current['name']] = current
        del current['name']
        del current['x']

    coord.request_stop()

    # Save tree
    print 'Writing tree to \'tree.pkl\''
    with open('tree.pkl', 'w') as f:
        pickle.dump(tree, f)

    coord.join(threads)
