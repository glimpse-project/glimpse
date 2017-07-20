#!/usr/bin/env python3

# Points for optimisation
# - [DONE] Asynchronous image decoding in queue, rather than in graph
# - [DONE as much as possible?] Remove, or reduce usage of feed_dict
# - Pre-process images before starting (decode, resize?)
# - [DONE] Process multiple images (or nodes?) at once
# - [DONE] Use the calculated label histograms to short-circuit combination
#   testing in graph (i.e. if n_labels < 2, skip testing)

# Optimisation points considered
# - All of these probably can't work because of IO constraints (not practical
#   to marshal them to/from the graph and cost of recalculation is low)
#   - Carry over shannon entropy calculations from parent node
#   - Carry over label histogram calculations from parent node
#   - Don't recalculate the label histogram for the current node on every epoch

import os
import cv2
import sys
import math
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime

# Image dimensions
WIDTH=540
HEIGHT=960
# Pixels per meter
PPM=579.0
# Maximum number of images to process at once
BATCH_SIZE=20
# Number of images to pre-load in the queue
QUEUE_BUFFER=30
# Number of threads to use when pre-loading queue
QUEUE_THREADS=1
# Limit the number of images in the training set
DATA_LIMIT=0
# Number of epochs to train per node
N_EPOCHS=10
# Maximum number of u,v pairs to test per epoch. The paper specifies testing
# 2000 candidate u,v pairs (the equivalent number is N_EPOCHS * COMBO_SIZE)
COMBO_SIZE=50
# Target gain to move onto the next node
TARGET_GAIN=0.25
# How frequently to display epoch progress
DISPLAY_STEP=1
# Whether to display epoch progress
DISPLAY_EPOCH_PROGRESS=True
# Depth to train tree to (paper specifies 20)
MAX_DEPTH=20
# The range to probe for generated u/v pairs (paper specifies 129 pixel meters)
RANGE_UV = 1.29 * PPM
# Threshold range to test offset parameters with (paper specifies testing 50)
# (nb: Perhaps they searched, rather than testing a range? Or didn't distribute
#      linearly over the range?)
RANGE_T = 1.29
MIN_T = -RANGE_T/2.0
MAX_T = RANGE_T/2.0
N_T = 50
T_INC = RANGE_T / (float(N_T) - 0.5)
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
    exr = cv2.imdecode(np.frombuffer(data, dtype='uint8'), cv2.IMREAD_UNCHANGED)
    return exr[...,0:1]

def elapsed(begin):
    now = datetime.utcnow()
    seconds = (now - begin).total_seconds()
    minutes = seconds / 60
    hours = minutes / 60
    seconds = seconds % 60
    minutes = minutes % 60
    return hours, minutes, seconds

def get_offset_indices(image, pixels, x, u, v):
    # Calculate the two inner terms of equation (1) from 3.2 of the paper
    # for each candidate pixel.
    extents = tf.shape(image)[0:2] - 1
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
    uindices = tf.clip_by_value(tf.cast(x + u, tf.int32), 0, clip)
    vindices = tf.clip_by_value(tf.cast(x + v, tf.int32), 0, clip)

    return uindices, vindices

def shannon_entropy(values):
    y, idx, count = tf.unique_with_counts(values)
    ncount = tf.cast(count, tf.float32) / tf.cast(tf.reduce_sum(count), tf.float32)
    return -tf.reduce_sum(ncount * (tf.log(ncount) / tf.log(2.0))), y, ncount

def computeDepthPixels(depth_image, u, v, x):
    # Gather the candidate pixels from the depth image and apply
    # equation (1) from 3.2 of the paper
    depth_pixels = tf.gather_nd(depth_image, x)
    uindices, vindices = get_offset_indices(depth_image, depth_pixels, x, u, v)
    return tf.gather_nd(depth_image, uindices) - tf.gather_nd(depth_image, vindices)

def computeLabelHistogramAndEntropy(label_pixels):
    # Compute the shannon entropy for storage of the labels of the candidate
    # depth pixels.
    hq, x_labels, x_label_prob = shannon_entropy(label_pixels)
    q = tf.cast(tf.size(label_pixels), tf.float32)

    return hq, q, x_labels, x_label_prob

def computeGainAndLRIndices(fdepth_pixels, label_pixels, t, hq, q):
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

    # Short-circuit these equations if lindex/rindex are 0-length
    G = tf.cond(tf.size(lindex) < 1, \
        lambda: tf.constant(0.0), \
        lambda: tf.cond(tf.size(rindex) < 1, \
                        lambda: tf.constant(0.0), \
                        lambda: hq - ((ql / q * hql) + (qr / q * hqr))))

    # Return the indices for the left branch and the right branch, as well as
    # the calculated gain. These will be used later if this threshold and
    # candidate u,v pair produce the best accumulated gain over all images.
    return lindex, rindex, G

def testImage(depth_image, label_image, u, v, x, label_pixels, hq, q):
    # Compute the set of pixels that will be used for partitioning
    fdepth_pixels = computeDepthPixels(depth_image, u, v, x)

    def testThreshold(t, meta, meta_indices, lindices, rindices):
        lindex, rindex, G = computeGainAndLRIndices(fdepth_pixels, label_pixels, t, hq, q)

        # Log threshold and gain
        meta = tf.concat([meta, [G]], axis=0)

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

        return t + T_INC, meta, meta_indices, lindices, rindices

    _, meta, meta_indices, lindices, rindices = tf.while_loop(\
        lambda t, meta, meta_indices, lindices, rindices: t <= MAX_T, \
        testThreshold, \
        [tf.constant(MIN_T), tf.zeros([0], dtype=tf.float32), \
         tf.zeros([0, 2, 2], dtype=tf.int32), \
         tf.zeros([0], dtype=tf.int32), tf.zeros([0], dtype=tf.int32)], \
        shape_invariants=[tf.TensorShape([]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None, 2, 2]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None])])

    return meta, meta_indices, lindices, rindices

# Collect training data
print('Collecting training data...')
label_images = []
depth_images = []
for imagefile in find_files('training/color', ['png']):
    label_images.append(imagefile)
for imagefile in find_files('training/depth', ['exr']):
    depth_images.append(imagefile)

print('Sorting training data...')
label_images.sort()
depth_images.sort()
if DATA_LIMIT > 0:
    label_images = label_images[0:DATA_LIMIT]
    depth_images = depth_images[0:DATA_LIMIT]

n_images = len(label_images)
assert(n_images == len(depth_images))
print('%d training image sets' % (n_images))

print('Creating data reading nodes...')
# Setup file readers
label_files = tf.train.string_input_producer(label_images, shuffle=False)
depth_files = tf.train.string_input_producer(depth_images, shuffle=False)
reader = tf.WholeFileReader()

# Setup image loaders
depth_key, depth_value = reader.read(depth_files)
label_key, label_value = reader.read(label_files)

depth_image = tf.py_func(read_depth, [depth_value], tf.float32, stateful=False)
label_image = tf.cast(tf.image.decode_png(label_value, channels=1), tf.int32)

image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, \
                           dtypes=(tf.float32, tf.int32))
enqueue = image_queue.enqueue((depth_image, label_image))

qr = tf.train.QueueRunner(image_queue, [enqueue] * QUEUE_THREADS)
tf.train.add_queue_runner(qr)

##############################

print('Creating graph...')
# Number of iterations to run (number of images to test)
batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

# Number of u,v pairs to test
combo_size = tf.placeholder(tf.int32, shape=[], name='combo_size')

# Pick splitting candidates (1)
all_u = tf.placeholder(tf.float32, shape=[None, 2], name='all_u')
all_v = tf.placeholder(tf.float32, shape=[None, 2], name='all_v')

# Pixel coordinates to test
all_x = tf.placeholder(tf.int32, shape=[None, None, 2], name='pixels')
len_x = tf.placeholder(tf.int32, shape=[None], name='dim_pixels')

def collect_results(i, all_meta, all_meta_indices, all_lindices, \
                    all_rindices, all_n_labels, \
                    all_x_labels, all_x_label_probs):
    # Read in depth and label image
    depth_image, label_image = image_queue.dequeue()
    depth_image.set_shape([HEIGHT, WIDTH, 1])
    label_image.set_shape([HEIGHT, WIDTH, 3])

    # Slice out the appropriate x values for the image
    x = tf.slice(all_x, [i, 0, 0], [1, len_x[i], 2])[0]

    # Compute label pixels, histogram shannon entropy
    label_pixels = tf.reshape(tf.gather_nd(label_image, x), [len_x[i]])
    hq, q, x_labels, x_label_prob = computeLabelHistogramAndEntropy(label_pixels)

    # Short-circuit all this work if there are fewer than 2 labels (and thus
    # no gain increase is possible)
    loop_combo_size = tf.cond(tf.size(x_labels) < 2, \
                              lambda: tf.constant(0), \
                              lambda: combo_size)

    # Test u,v pairs against a range of thresholds
    def test_uv(i, all_meta, all_meta_indices, all_lindices, all_rindices):
        # Slice out the appropriate u and v
        u = tf.slice(all_u, [i, 0], [1, 2])[0]
        v = tf.slice(all_v, [i, 0], [1, 2])[0]

        # Run an iteration
        meta, meta_indices, lindices, rindices = \
            testImage(depth_image, label_image, u, v, x, label_pixels, hq, q)

        # Collect the run metadata
        # meta is [N_T], float32
        #   (gain)
        all_meta = tf.concat([all_meta, [meta]], axis=0)

        # meta_indices is [N_T,2,2], int32
        #   (pairs of indices into lindices and rindices per threshold)
        all_meta_indices = tf.concat([all_meta_indices, [meta_indices]], axis=0)

        # lindices and rindices are [?], int32
        #   (indices into x for left and right branches, per threshold)
        all_lindices = tf.concat([all_lindices, lindices], axis=0)
        all_rindices = tf.concat([all_rindices, rindices], axis=0)

        return i + 1, all_meta, all_meta_indices, all_lindices, \
            all_rindices

    _i, all_meta, all_meta_indices, all_lindices, all_rindices = \
        tf.while_loop( \
            lambda i, a, b, c, d: i < loop_combo_size, \
            test_uv, \
            [0, all_meta, all_meta_indices, all_lindices, all_rindices], \
            shape_invariants=[tf.TensorShape([]), \
                              tf.TensorShape([None, N_T]), \
                              tf.TensorShape([None, N_T, 2, 2]), \
                              tf.TensorShape([None]), \
                              tf.TensorShape([None])])

    # Collect label histogram data
    # xlabels is [?], int32
    #   (Unique labels for pixels in x)
    all_x_labels = tf.concat([all_x_labels, x_labels], axis=0)

    # x_label_prob is [len(xlabels)], float32
    #   (Distribution of each label in xlabels)
    all_x_label_probs = tf.concat([all_x_label_probs, x_label_prob], axis=0)

    # Store the size of the label histogram for later indexing
    all_n_labels = tf.concat([all_n_labels, [tf.size(x_labels)]], axis=0)

    return i + 1, all_meta, all_meta_indices, all_lindices, \
        all_rindices, all_n_labels, all_x_labels, all_x_label_probs

all_meta = tf.zeros([0, N_T], dtype=tf.float32)
all_meta_indices = tf.zeros([0, N_T, 2, 2], dtype=tf.int32)
all_lindices = tf.zeros([0], dtype=tf.int32)
all_rindices = tf.zeros([0], dtype=tf.int32)
all_n_labels = tf.zeros([0], dtype=tf.int32)
all_x_labels = tf.zeros([0], dtype=tf.int32)
all_x_label_probs = tf.zeros([0], dtype=tf.float32)

# Run batch_size iterations over combo_size u,v pairs
_i, all_meta, all_meta_indices, all_lindices, all_rindices, all_n_labels, \
    all_x_labels, all_x_label_probs = tf.while_loop( \
        lambda i, a, b, c, d, e, f, g: i < batch_size, \
        collect_results, \
        [0, all_meta, all_meta_indices, all_lindices, \
         all_rindices, all_n_labels, all_x_labels, all_x_label_probs], \
        shape_invariants=[tf.TensorShape([]), \
                          tf.TensorShape([None, N_T]), \
                          tf.TensorShape([None, N_T, 2, 2]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None])])

# Initialise and run the session
init = tf.global_variables_initializer()
session = tf.Session()

with session.as_default():
    print('Initialising...')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    session.run(init)

    # Generate the coordinates for the pixel samples for the root node
    print('Generating initial coordinates...')
    initial_coords = \
        np.stack((np.random.random_integers(0, HEIGHT-1, N_SAMP * n_images), \
                  np.random.random_integers(0, WIDTH-1, N_SAMP * n_images)), \
                 axis=1)

    # Push the root node onto the training queue
    queue = [{ 'name': '', \
               'x': np.reshape(initial_coords, (n_images, N_SAMP, 2)),
               'xl': np.tile([N_SAMP], n_images)}]

    # Initialise the output tree
    tree = {}

    # Start timing
    begin = datetime.utcnow()

    while len(queue) > 0:
        current = queue.pop(0)
        best_G = None
        best_c = None
        best_u = None
        best_v = None
        best_t = None
        best_label_probs = None
        best_isNotLeaf = False
        nextNodes = []

        # Pixel coordinates are padded, so we can just look at the first entry
        # to work out the largest amount of pixels in any image
        max_pixels = len(current['x'][0])

        # If this is the deepest node in the tree, or we don't have pixels
        # to check, don't waste time calculating u,v pairs and their results.
        force_leaf = False
        if len(current['name']) >= MAX_DEPTH or max_pixels < 2:
            force_leaf = True

        print('Training node (%s) (Max %d pixels)' % \
              (current['name'], max_pixels))

        for epoch in range(1, N_EPOCHS + 1):
            if epoch == 1 or \
               epoch % DISPLAY_STEP == 0 or \
               epoch == N_EPOCHS:
                hours, minutes, seconds = elapsed(begin)
                print('\t(%s) Epoch %dx%d (%d) (%02d:%02d:%02d elapsed)' % \
                      (current['name'], epoch, COMBO_SIZE, epoch * COMBO_SIZE, \
                       hours, minutes, seconds))

            # Initialise trial splitting candidates
            c_u = np.random.uniform(-RANGE_UV/2.0, RANGE_UV/2.0, (COMBO_SIZE, 2))
            c_v = np.random.uniform(-RANGE_UV/2.0, RANGE_UV/2.0, (COMBO_SIZE, 2))

            labels = []
            gains = {}
            thresholds = np.linspace(MIN_T, MAX_T, N_T)
            for i in range(COMBO_SIZE):
                for t in thresholds:
                    key = (t, i)
                    gains[key] = { 'g': np.zeros([n_images], dtype=np.float32), \
                                   'lcoords': [[]] * n_images, \
                                   'rcoords': [[]] * n_images,
                                   'nlcoords': np.zeros([n_images], dtype=np.int32),
                                   'nrcoords': np.zeros([n_images], dtype=np.int32) }

            batch = 0
            while batch < n_images:
                if DISPLAY_EPOCH_PROGRESS:
                    hours, minutes, seconds = elapsed(begin)
                    sys.stdout.write( \
                        '(%02d:%02d:%02d) Epoch %04d Training batch %04d/%04d... ' % \
                        (hours, minutes, seconds, epoch, batch / BATCH_SIZE + 1, \
                         (n_images - 1) / BATCH_SIZE + 1))
                    sys.stdout.flush()

                c_batch_size = min(n_images - batch, BATCH_SIZE)

                # Work out if we can skip gain calculation for this batch
                c_combo_size = COMBO_SIZE
                if force_leaf:
                    c_combo_size = 0
                else:
                    can_skip = True
                    for i in range(batch, batch + c_batch_size):
                        if current['xl'][i] > 1:
                            can_skip = False
                            break
                    if can_skip:
                        c_combo_size = 0

                t_meta, t_meta_indices, t_lindices, t_rindices, t_n_labels, \
                    t_labels, t_label_probs = \
                        session.run((all_meta, all_meta_indices, all_lindices, \
                                     all_rindices, all_n_labels, all_x_labels, \
                                     all_x_label_probs), \
                                    feed_dict={all_u: c_u, \
                                               all_v: c_v, \
                                               all_x: current['x'][batch:(batch+c_batch_size)], \
                                               len_x: current['xl'][batch:(batch+c_batch_size)], \
                                               batch_size: c_batch_size, \
                                               combo_size: c_combo_size})

                if DISPLAY_EPOCH_PROGRESS:
                    hours, minutes, seconds = elapsed(begin)
                    sys.stdout.write('(%02d:%02d:%02d) Collating results...\r' % \
                                     (hours, minutes, seconds))
                    sys.stdout.flush()

                # Collect left/right pixels for each u,v,t combination and
                # label histograms for each image in this batch.
                lindex_base = 0
                rindex_base = 0
                label_base = 0
                idx = 0

                for i in range(batch, batch + c_batch_size):
                    # Skip processing if there are no pixels
                    if current['xl'][i] == 0:
                        continue

                    # Store the label histogram for this image
                    label_end = label_base + t_n_labels[i - batch]
                    labels.append(list(zip(t_labels[label_base:label_end], \
                                           t_label_probs[label_base:label_end])))
                    label_base = label_end

                    # Skip processing if there's only one label
                    if len(labels[-1]) < 2:
                        continue

                    for j in range(c_combo_size):
                        meta = t_meta[idx]
                        meta_indices = t_meta_indices[idx]
                        idx += 1

                        assert(len(meta) == N_T)
                        for k in range(N_T) :
                            t = thresholds[k]
                            t_g = meta[k]

                            if math.isnan(t_g) or t_g < 0.0:
                                t_g = 0.0

                            lstart = lindex_base + meta_indices[k][0][0]
                            lend = lindex_base + meta_indices[k][0][1]
                            rstart = rindex_base + meta_indices[k][1][0]
                            rend = rindex_base + meta_indices[k][1][1]

                            tt_lindices = t_lindices[lstart:lend]
                            tt_rindices = t_rindices[rstart:rend]

                            t_lcoords = current['x'][i][tt_lindices]
                            t_rcoords = current['x'][i][tt_rindices]

                            assert(len(t_lcoords) == lend - lstart), \
                                't_lcoords malformed (%d != %d)' % \
                                (len(t_lcoords), (lend - lstart))
                            assert(len(t_rcoords) == rend - rstart), \
                                't_rcoords malformed (%d != %d)' % \
                                (len(t_rcoords), (rend - rstart))

                            key = (t, j)
                            gains[key]['g'][i] = t_g
                            gains[key]['lcoords'][i] = t_lcoords
                            gains[key]['rcoords'][i] = t_rcoords
                            gains[key]['nlcoords'][i] = len(t_lcoords)
                            gains[key]['nrcoords'][i] = len(t_rcoords)

                        lindex_base += meta_indices[-1][0][1]
                        rindex_base += meta_indices[-1][1][1]

                # Progress to the next batch
                batch += c_batch_size
                if DISPLAY_EPOCH_PROGRESS:
                    sys.stdout.write('                                     ' + \
                                     '                                     ' + \
                                     '       \r')
                    sys.stdout.flush()

            # Accumulate threshold gains to find the most effective 't' for
            # this u,v pair.
            gain = -1
            threshold = -1
            combo = -1

            for key, gain_data in gains.items():
                acc_gain = np.sum(np.array(gain_data['g']))
                if acc_gain > gain:
                    gain = acc_gain
                    threshold = key[0]
                    combo = key[1]

            gain /= n_images

            if best_G is None or gain > best_G:
                best_label_probs = labels

                # Because we short-circuit everything except label calculation
                # on the last node of the tree, these values can be unset
                if gain != -1:
                    best_G = gain
                    best_u = c_u[combo]
                    best_v = c_v[combo]
                    best_t = threshold

                    # Collect l/r pixels for this u,v,t combination
                    gain_data = gains[(threshold, combo)]
                    lcoords = gain_data['lcoords']
                    rcoords = gain_data['rcoords']
                    nlcoords = gain_data['nlcoords']
                    nrcoords = gain_data['nrcoords']
                    maxlcoords = np.amax(nlcoords)
                    maxrcoords = np.amax(nrcoords)

                    # Pad out (lr)coords
                    for i in range(len(lcoords)):
                        lcoords[i] = \
                            np.resize(np.array(lcoords[i], dtype=np.int32), \
                                      (maxlcoords, 2))
                    for i in range(len(rcoords)):
                        rcoords[i] = \
                            np.resize(np.array(rcoords[i], dtype=np.int32), \
                                      (maxrcoords, 2))

                    # Add left/right nodes to the queue
                    nextNodes = [{ 'name': current['name'] + 'l', \
                                   'x': lcoords, \
                                   'xl': nlcoords }, \
                                 { 'name': current['name'] + 'r', \
                                   'x': rcoords, \
                                   'xl': nrcoords }]

                    print('\t\tG = ' + str(best_G))
                    print('\t\tu = ' + str(best_u))
                    print('\t\tv = ' + str(best_v))
                    print('\t\tt = ' + str(best_t))
                    print('\t\tl,r = %d, %d' % (maxlcoords, maxrcoords))

            # Don't process further epochs if this is a leaf node or we've
            # reached our target gain
            if force_leaf or best_G >= TARGET_GAIN:
                break

        # If gain hasn't increased for any images, there's no use in going
        # down this branch further.
        # TODO: Maybe this should be a threshold rather than just zero.
        if best_G is not None and \
           best_G > 0.0 and \
           force_leaf is not True:
            # Store the trained u,v,t parameters
            current['u'] = best_u
            current['v'] = best_v
            current['t'] = best_t

            # Add the next nodes to train to the queue
            queue.extend(nextNodes)
        else:
            # This is a leaf node, store the label probabilities
            excuse = None
            if len(current['name']) >= MAX_DEPTH:
                excuse = 'Maximum depth (%d) reached' % (MAX_DEPTH)
            elif max_pixels < 2:
                excuse = 'No pixels left to separate'
            elif best_G <= 0.0:
                excuse = 'No gain increase'

            print('\tLeaf node (%s):' % (excuse))
            label_probs = {}
            for probs in best_label_probs:
                for prob in probs:
                    key = prob[0]
                    if key not in label_probs:
                        label_probs[key] = prob[1]
                    else:
                        label_probs[key] += prob[1]
            for key, value in label_probs.items():
                label_probs[key] = value / float(len(best_label_probs))
                print('\t\t%8d - %0.3f' % (key, label_probs[key]))

            current['label_probs'] = label_probs

        # Save finished node to tree
        tree[current['name']] = current

        # Remove unneeded data from the node
        del current['name']
        del current['x']
        del current['xl']

    coord.request_stop()

    # Save tree
    print('Writing tree to \'tree.pkl\'')
    with open('tree.pkl', 'wb') as f:
        pickle.dump(tree, f)

    coord.join(threads)
