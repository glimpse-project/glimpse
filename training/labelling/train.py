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
# Number of epochs to train per node (paper specifies 2000)
N_EPOCHS=2000
# Depth to train tree to (paper specifies 20)
MAX_DEPTH=20
# Maximum probe offset for random u/v values, in pixel meters (paper specifies 129)
MAX_UV = 129
# Threshold range to test offset parameters with (paper specifies testing 50)
# (nb: Perhaps they searched, rather than testing a range?)
MIN_T = -0.5
MAX_T = 0.5
N_T = 49.0
# Number of pixel samples (paper specifies 2000)
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
    # Calculate the two inner terms of equation (1) from 3.2 of the paper
    # for each candidate pixel.
    extents = tf.subtract(tf.shape(image)[0:2], 1)
    n_pixels = tf.size(pixels)
    clip = tf.reshape(tf.tile(extents, [n_pixels]), [n_pixels, 2])
    pixels = tf.cast(tf.reshape(pixels, [n_pixels]), tf.float32)

    ux = tf.divide(tf.tile([u[0]], [n_pixels]), pixels)
    uy = tf.divide(tf.tile([u[1]], [n_pixels]), pixels)
    u = tf.stack((ux, uy), axis=1)

    vx = tf.divide(tf.tile([v[0]], [n_pixels]), pixels)
    vy = tf.divide(tf.tile([v[1]], [n_pixels]), pixels)
    v = tf.stack((vx, vy), axis=1)

    # Note, we clip the coordinates here. The paper says that any coordinate
    # that references outside of the image should essentially result in the
    # background depth - we can assure this by processing the input images.
    x = tf.cast(x, tf.float32)
    uindices = tf.clip_by_value(tf.cast(tf.add(x, u), tf.int32), 0, clip)
    vindices = tf.clip_by_value(tf.cast(tf.add(x, v), tf.int32), 0, clip)

    return uindices, vindices

def decolorize(pixels):
    pixels = tf.multiply(tf.cast(pixels, tf.int32), [256 * 256, 256, 1])
    return tf.reduce_sum(pixels, 1)

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

def computeDepthPixels(depth_image, u, v, x):
    # Gather the candidate pixels from the depth image and apply
    # equation (1) from 3.2 of the paper
    depth_pixels = tf.gather_nd(depth_image, x)
    uindices, vindices = get_offset_indices(depth_image, depth_pixels, x, u, v)
    return tf.gather_nd(depth_image, uindices) - tf.gather_nd(depth_image, vindices)

def getLabelPixels(label_image, x):
    # Gather and compress the RGB label image
    return decolorize(tf.gather_nd(label_image, x))

def computeLabelHistogramAndEntropy(label_pixels):
    # Compute the shannon entropy for storage of the labels of the candidate
    # depth pixels.
    hq, x_labels, x_label_prob = shannon_entropy(label_pixels)
    q = tf.cast(tf.size(label_pixels), tf.float32)

    return hq, q, x_labels, x_label_prob

def computeGainAndLRIndices(fdepth_pixels, label_pixels, t):
    # Partition candidate depth pixels into two groups
    # TODO: Find an alternative way of doing this that doesn't require two
    #       conditions to be evaluated.
    lindex, _ = tf.split(tf.cast(tf.where(tf.less(fdepth_pixels, t)), tf.int32), 2, axis=1)
    rindex, _ = tf.split(tf.cast(tf.where(tf.greater_equal(fdepth_pixels, t)), tf.int32), 2, axis=1)

    # Compute gain (see equation (6) from 3.3 of the paper)
    llabel_pixels = tf.gather_nd(label_pixels, lindex)
    hql, l_labels, l_label_prob = shannon_entropy(llabel_pixels)
    ql = tf.cast(tf.shape(llabel_pixels)[0], tf.float32)

    rlabel_pixels = tf.gather_nd(label_pixels, rindex)
    hqr, r_labels, r_label_prob = shannon_entropy(rlabel_pixels)
    qr = tf.cast(tf.shape(rlabel_pixels)[0], tf.float32)

    G = hq - ((ql / q * hql) + (qr / q * hqr))

    # Return the indices for the left branch and the right branch, as well as
    # the calculated gain. These will be used later if this threshold and
    # candidate u,v pair produce the best accumulated gain over all images.
    return lindex, rindex, G

# Read in depth and label image
depth_image, label_image = image_queue.dequeue()
depth_image.set_shape([HEIGHT, WIDTH, 1])
label_image.set_shape([HEIGHT, WIDTH, 3])

# Pick splitting candidates (1)
u = tf.placeholder(tf.float32, shape=[2], name='u')
v = tf.placeholder(tf.float32, shape=[2], name='v')

# Pixel coordinates to test
x = tf.placeholder(tf.int32, shape=[None, 2], name='x')

# Compute the set of pixels that will be used for partitioning
fdepth_pixels = computeDepthPixels(depth_image, u, v, x)

# Compute label histogram and shannon entropy
label_pixels = getLabelPixels(label_image, x)
hq, q, x_labels, x_label_prob = \
        computeLabelHistogramAndEntropy(label_pixels)

t_inc = (MAX_T - MIN_T) / (N_T - 1)
def testThreshold(t, meta, meta_indices, lindices, rindices):
    lindex, rindex, G = computeGainAndLRIndices(fdepth_pixels, label_pixels, t)

    # Log threshold and gain
    meta = tf.concat([meta, [[t, G]]], axis=0)

    # Work out index ranges
    lstart = tf.size(lindices)
    rstart = tf.size(rindices)
    lsize = tf.size(lindex)
    rsize = tf.size(rindex)
    lrange = [lstart, lstart + lsize]
    rrange = [rstart, rstart + rsize]
    meta_index = [lrange, rrange]

    # Flatten indices
    lindex = tf.reshape(lindex, [lsize])
    rindex = tf.reshape(rindex, [rsize])

    # Add indices to index lists
    meta_indices = tf.concat([meta_indices, [meta_index]], axis=0)
    lindices = tf.concat([lindices, lindex], axis=0)
    rindices = tf.concat([rindices, rindex], axis=0)

    return tf.add(t, t_inc), meta, meta_indices, lindices, rindices

_, meta, meta_indices, lindices, rindices = tf.while_loop(\
    lambda t, meta, meta_indices, lindices, rindices: t <= MAX_T, \
    testThreshold, \
    [tf.constant(MIN_T), tf.zeros([0, 2], dtype=tf.float32), \
     tf.zeros([0, 2, 2], dtype=tf.int32), \
     tf.zeros([0], dtype=tf.int32), tf.zeros([0], dtype=tf.int32)], \
    shape_invariants=[tf.TensorShape([]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None, 2, 2]), \
                      tf.TensorShape([None]), \
                      tf.TensorShape([None])])

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

            labels = []
            gains = {}
            for i in range(n_images):
                # Bail out if we have no pixels to check
                if len(current['x'][i]) < 2:
                    continue

                t_meta, t_meta_indices, t_lindices, t_rindices, t_labels, t_label_probs = \
                    session.run((meta, meta_indices, lindices, rindices, \
                                 x_labels, x_label_prob), \
                                feed_dict={u: c_u, v: c_v, x: current['x'][i]})

                # Store all the thresholds and associated gains and indices
                # for analysis after this epoch finishes
                for j in range(len(t_meta)):
                    t = t_meta[j][0]
                    t_g = t_meta[j][1]
                    tt_lindices = t_lindices[t_meta_indices[j][0][0]:t_meta_indices[j][0][1]]
                    tt_rindices = t_rindices[t_meta_indices[j][1][0]:t_meta_indices[j][1][1]]
                    t_lcoords = current['x'][i][tt_lindices]
                    t_rcoords = current['x'][i][tt_rindices]

                    if t not in gains:
                        gains[t] = []

                    gains[t].append({ 'g': t_g, 'lcoords': t_lcoords, 'rcoords': t_rcoords })

                labels.append(np.stack((t_labels, t_label_probs), axis=1))

            # Accumulate threshold gains to find the most effective 't' for
            # this u,v pair.
            gain = -1
            threshold = -1
            lcoords = None
            rcoords = None
            for t, gain_data in gains.items():
                cum_gain = 0
                cum_lcoords = []
                cum_rcoords = []
                for datum in gain_data:
                    cum_gain += datum['g']
                    cum_lcoords.append(datum['lcoords'])
                    cum_rcoords.append(datum['rcoords'])
                if cum_gain > gain:
                    gain = cum_gain
                    threshold = t
                    lcoords = cum_lcoords
                    rcoords = cum_rcoords

            gain /= n_images

            if best_G is None or gain > best_G:
                best_G = gain
                best_u = c_u
                best_v = c_v
                best_t = threshold
                best_label_probs = labels
                nextNodes = [{ 'name': current['name'] + 'l', \
                               'x': lcoords }, \
                             { 'name': current['name'] + 'r', \
                               'x': rcoords }]
                print '\tG = ' + str(best_G)
                print '\tt = ' + str(best_t)
                print '\tu = ' + str(best_u)
                print '\tv = ' + str(best_v)

        # Add the next nodes to train to the queue
        # TODO: Is the logic here correct?
        if len(nextNodes[0]['x']) > 0 and \
           len(nextNodes[1]['x']) > 0 and \
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