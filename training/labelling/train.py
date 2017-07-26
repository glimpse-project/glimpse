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
import time
import pickle
import signal
import numpy as np
import multiprocessing
import tensorflow as tf

# Image dimensions
WIDTH=540
HEIGHT=960
# Pixels per meter
PPM=579.0
# Number of images to pre-load in the queue
QUEUE_BUFFER=1000
# Number of threads to use when pre-loading queue
QUEUE_THREADS=multiprocessing.cpu_count()
# Limit the number of images in the training set
DATA_LIMIT=0
# Number of epochs to train per node
N_EPOCHS=1
# Maximum number of nodes to train at once
MAX_NODES=16384
# Maximum number of u,v pairs to test per epoch. The paper specifies testing
# 2000 candidate u,v pairs (the equivalent number is N_EPOCHS * COMBO_SIZE)
COMBO_SIZE=2000
# How frequently to display epoch progress
DISPLAY_STEP=1
# How long to let elapse before creating a checkpoint (in seconds)
CHECKPOINT_TIME=1800
# Whether to display verbose progress
DISPLAY_VERBOSE_PROGRESS=False
# Depth to train tree to (paper specifies 20)
MAX_DEPTH=20
# The range to probe for generated u/v pairs (paper specifies 129 pixel meters)
RANGE_UV = 1.29 * PPM
MIN_UV = -RANGE_UV/2.0
MAX_UV = RANGE_UV/2.0
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
    now = time.monotonic()
    seconds = now - begin
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

def getLRPixelIndices(depth_pixels, t):
    # TODO: Find an alternative way of doing this that doesn't require two
    #       conditions to be evaluated.
    lindex, _ = tf.split(tf.cast(tf.where(tf.less(depth_pixels, t)), tf.int32), 2, axis=1)
    rindex, _ = tf.split(tf.cast(tf.where(tf.greater_equal(depth_pixels, t)), tf.int32), 2, axis=1)

    return lindex, rindex

def computeGain(fdepth_pixels, label_pixels, t, hq, q):
    # Partition candidate depth pixels into two groups
    lindex, rindex = getLRPixelIndices(fdepth_pixels, t)

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
    return G

def testImage(depth_image, u, v, x, label_pixels, hq, q):
    # Compute the set of pixels that will be used for partitioning
    fdepth_pixels = computeDepthPixels(depth_image, u, v, x)

    def testThreshold(t, meta):
        G = computeGain(fdepth_pixels, label_pixels, t, hq, q)

        # Log gain
        meta = tf.concat([meta, [G]], axis=0)

        return t + T_INC, meta

    _, meta = tf.while_loop(\
        lambda t, meta: t <= MAX_T, \
        testThreshold, \
        [MIN_T, tf.zeros([0], dtype=tf.float32)], \
        shape_invariants=[tf.TensorShape([]), \
                          tf.TensorShape([None])])

    return meta

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

depth_image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, dtypes=(tf.float32))
label_image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, dtypes=(tf.int32))
enqueue_depth = depth_image_queue.enqueue((depth_image))
enqueue_label = label_image_queue.enqueue((label_image))

depth_qr = tf.train.QueueRunner(depth_image_queue, [enqueue_depth] * QUEUE_THREADS)
label_qr = tf.train.QueueRunner(label_image_queue, [enqueue_label] * QUEUE_THREADS)
tf.train.add_queue_runner(depth_qr)
tf.train.add_queue_runner(label_qr)

##############################

print('Creating graph...')
# Number of nodes being tested on
nodes_size = tf.placeholder(tf.int32, shape=[], name='nodes_size')

# Pick splitting candidates (1)
all_u = tf.random_uniform([nodes_size, COMBO_SIZE, 2], MIN_UV, MAX_UV, name='all_u')
all_v = tf.random_uniform([nodes_size, COMBO_SIZE, 2], MIN_UV, MAX_UV, name='all_v')

# Pixel coordinates to test
all_x = tf.placeholder(tf.int32, shape=[None, n_images, None, 2], name='pixels')
len_x = tf.placeholder(tf.int32, shape=[None, n_images], name='dim_pixels')

# Which nodes to test
skip_mask = tf.placeholder(tf.bool, shape=[None], name='skip_mask')

def accumulate_gain(i, acc_gain, all_n_labels, all_x_labels, all_x_label_probs):
    # Read in depth and label image
    depth_image = depth_image_queue.dequeue()
    label_image = label_image_queue.dequeue()
    depth_image.set_shape([HEIGHT, WIDTH, 1])
    label_image.set_shape([HEIGHT, WIDTH, 1])

    def test_node(n, node_gains, all_x_labels, all_x_label_probs, all_n_labels):
        # Slice out the appropriate x values for the image
        x = tf.slice(all_x, [n, i, 0, 0], [1, 1, len_x[n][i], 2])[0][0]

        # Compute label pixels, histogram shannon entropy
        label_pixels = tf.reshape(tf.gather_nd(label_image, x), [len_x[n][i]])
        hq, q, x_labels, x_label_prob = computeLabelHistogramAndEntropy(label_pixels)

        # Test u,v pairs against a range of thresholds
        def test_uv(i, all_gain):
            # Slice out the appropriate u and v
            u = tf.slice(all_u, [n, i, 0], [1, 1, 2])[0][0]
            v = tf.slice(all_v, [n, i, 0], [1, 1, 2])[0][0]

            # Short-circuit all this work if there are fewer than 2 labels (and thus
            # no gain increase is possible)
            G = tf.cond(tf.size(x_labels) < 2, \
                        lambda: tf.zeros([N_T]), \
                        lambda: testImage(depth_image, u, v, x, label_pixels, hq, q))

            # Collect the run metadata
            # meta is [N_T], float32
            #   (gain)
            all_gain = tf.concat([all_gain, [G]], axis=0)

            return i + 1, all_gain

        _i, all_gain = \
            tf.while_loop( \
                lambda i, all_gain: i < COMBO_SIZE, \
                test_uv, \
                [0, tf.zeros([0, N_T])], \
                shape_invariants=[tf.TensorShape([]), \
                                  tf.TensorShape([None, N_T])])

        # Keep track of the gains for this node, but short-circuit if there are
        # no label pixels or we're not testing this node
        add_gain = tf.concat([node_gains, [all_gain]], axis=0)
        skip_gain = tf.concat([node_gains, tf.zeros([1, COMBO_SIZE, N_T], \
                                                    dtype=tf.float32)], axis=0)

        node_gains = tf.cond(skip_mask[n], \
            lambda: skip_gain,
            lambda: tf.cond(tf.size(x_labels) < 2, \
                            lambda: skip_gain,
                            lambda: add_gain))

        # Collect label histogram data
        # xlabels is [?], int32
        #   (Unique labels for pixels in x)
        all_x_labels = tf.concat([all_x_labels, x_labels], axis=0)

        # x_label_prob is [len(xlabels)], float32
        #   (Distribution of each label in xlabels)
        all_x_label_probs = tf.concat([all_x_label_probs, x_label_prob], axis=0)

        # Store the index of the label histogram
        all_n_labels = tf.concat([all_n_labels, [tf.size(x_labels)]], axis=0)

        return n+1, node_gains, all_x_labels, all_x_label_probs, all_n_labels

    _n, node_gains, all_x_labels, all_x_label_probs, all_n_labels = \
        tf.while_loop( \
            lambda n, _ng, _xl, _xlp, _nl: n < nodes_size,
            test_node,
            [0, tf.zeros([0, COMBO_SIZE, N_T]), \
             all_x_labels, all_x_label_probs, all_n_labels], \
            shape_invariants=[tf.TensorShape([]), \
                              tf.TensorShape([None, COMBO_SIZE, N_T]), \
                              tf.TensorShape([None]), \
                              tf.TensorShape([None]), \
                              tf.TensorShape([None])])

    # Accumulate gain
    acc_gain += node_gains

    return i + 1, acc_gain, all_n_labels, all_x_labels, all_x_label_probs

# Run n_images iterations over COMBO_SIZE u,v pairs
acc_gain = tf.zeros([nodes_size, COMBO_SIZE, N_T], dtype=tf.float32)
all_n_labels = tf.zeros([0], dtype=tf.int32)
all_x_labels = tf.zeros([0], dtype=tf.int32)
all_x_label_probs = tf.zeros([0], dtype=tf.float32)

_i, acc_gain, all_n_labels, all_x_labels, all_x_label_probs = tf.while_loop( \
    lambda i, a, b, c, d: i < n_images, \
    accumulate_gain, \
    [0, acc_gain, all_n_labels, all_x_labels, all_x_label_probs], \
    shape_invariants=[tf.TensorShape([]), \
                      tf.TensorShape([None, COMBO_SIZE, N_T]), \
                      tf.TensorShape([None]), \
                      tf.TensorShape([None]), \
                      tf.TensorShape([None])])

# Scan over all_n_labels to make the indices absolute
all_n_labels = tf.scan(lambda a, x: a + x, all_n_labels)

# Find the best gain and the best combination index
flat_gain = tf.reshape(acc_gain, [nodes_size, COMBO_SIZE * N_T])
def find_best_combinations(n, best_gains, best_u, best_v, best_t):
    idx = tf.cast(tf.argmax(flat_gain[n], axis=0), dtype=tf.int32)
    combo = tf.div(idx, N_T)
    threshold = tf.mod(idx, N_T)

    best_gains = tf.concat([best_gains, [flat_gain[n][idx]]], axis=0)
    best_u = tf.concat([best_u, [all_u[n][combo]]], axis=0)
    best_v = tf.concat([best_v, [all_v[n][combo]]], axis=0)
    best_t = tf.concat([best_t, [threshold]], axis=0)

    return n+1, best_gains, best_u, best_v, best_t

best_gains = tf.zeros([0], dtype=tf.float32)
best_u = tf.zeros([0, 2], dtype=tf.float32)
best_v = tf.zeros([0, 2], dtype=tf.float32)
best_t = tf.zeros([0], dtype=tf.int32)

_n, best_gains, best_u, best_v, best_t = tf.while_loop( \
    lambda n, _bg, _bu, _bv, _bt: n < nodes_size, \
    find_best_combinations, \
    [0, best_gains, best_u, best_v, best_t], \
    shape_invariants=[tf.TensorShape([]), \
                      tf.TensorShape([None]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None])])

# Construct graph for retrieving lr pixel coordinates
# For retrieving the coordinates of a particular u,v,t combination
u = tf.placeholder(tf.float32, shape=[None, 2], name='u')
v = tf.placeholder(tf.float32, shape=[None, 2], name='v')
t = tf.placeholder(tf.float32, shape=[None], name='t')

def collect_indices(i, all_meta_indices, all_lindices, all_rindices):
    # Dequeue the depth image
    depth_image = depth_image_queue.dequeue()
    depth_image.set_shape([HEIGHT, WIDTH, 1])

    def collect_node_indices(n, all_meta_indices, all_lindices, all_rindices):
        # Slice out the appropriate x values for the image
        x = tf.slice(all_x, [n, i, 0, 0], [1, 1, len_x[n][i], 2])[0][0]

        # Compute the depth pixels that we'll need to compare against the threshold
        depth_pixels = computeDepthPixels(depth_image, u[n], v[n], x)

        # Get the left/right pixel indices
        lindices, rindices = getLRPixelIndices(depth_pixels, t[n])

        # Work out index ranges
        lsize = tf.size(lindices)
        rsize = tf.size(rindices)
        meta_index = [lsize, rsize]

        # Flatten indices
        lindices = tf.reshape(lindices, [lsize])
        rindices = tf.reshape(rindices, [rsize])

        # Add indices to index lists
        all_meta_indices = tf.concat([all_meta_indices, [meta_index]], axis=0)
        all_lindices = tf.concat([all_lindices, lindices], axis=0)
        all_rindices = tf.concat([all_rindices, rindices], axis=0)

        return n+1, all_meta_indices, all_lindices, all_rindices

    _n, all_meta_indices, all_lindices, all_rindices = tf.while_loop( \
        lambda n, _ami, _al, _ar: n < nodes_size, \
        collect_node_indices, \
        [0, all_meta_indices, all_lindices, all_rindices],
        shape_invariants=[tf.TensorShape([]), \
                          tf.TensorShape([None, 2]), \
                          tf.TensorShape([None]), \
                          tf.TensorShape([None])])

    return i + 1, all_meta_indices, all_lindices, all_rindices

all_meta_indices = tf.zeros([0, 2], dtype=tf.int32)
all_lindices = tf.zeros([0], dtype=tf.int32)
all_rindices = tf.zeros([0], dtype=tf.int32)

_i, all_meta_indices, all_lindices, all_rindices = tf.while_loop( \
    lambda i, a, b, c: i < n_images, \
    collect_indices, \
    [0, all_meta_indices, all_lindices, all_rindices], \
    shape_invariants=[tf.TensorShape([]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None]), \
                      tf.TensorShape([None])])

# Initialise and run the session
init = tf.global_variables_initializer()
session = tf.Session()

def write_checkpoint(root, queue):
    checkpoint = (root, queue)
    with open('checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

def read_checkpoint():
    with open('checkpoint.pkl', 'rb') as f:
        print('Restoring checkpoint...')
        checkpoint = pickle.load(f, encoding='latin1')
        return checkpoint

class BreakHandler:
    triggered = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, signal, frame):
        self.triggered = True

with session.as_default():
    print('Initialising...')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    session.run(init)

    try:
        # Try to restore checkpoint
        root, queue = read_checkpoint()
    except FileNotFoundError:
        # Generate the coordinates for the pixel samples for the root node
        print('Generating initial coordinates...')
        initial_coords = \
            np.stack((np.random.random_integers(0, HEIGHT-1, N_SAMP * n_images), \
                      np.random.random_integers(0, WIDTH-1, N_SAMP * n_images)), \
                     axis=1)

        # Push the root node onto the training queue
        root = { 'name': '', \
                 'depth': 0,
                 'x': np.reshape(initial_coords, (n_images, N_SAMP, 2)),
                 'xl': np.tile([N_SAMP], n_images)}
        queue = [root]

    # Initialise the output tree
    thresholds = np.linspace(MIN_T, MAX_T, N_T)

    # Start timing
    begin = time.monotonic()
    checkpoint_time = begin

    # Setup signal handler
    exit_after_checkpoint = BreakHandler()

    while len(queue) > 0:
        n_nodes = min(len(queue), MAX_NODES)
        hours, minutes, seconds = elapsed(begin)
        print('(%02d:%02d:%02d) Gathering data for %d nodes' % \
              (hours, minutes, seconds, n_nodes))

        # Create variables to keep track of the best u,v,t combinations per node
        candidate_G = np.full([n_nodes], -1.0, dtype=np.float32)
        candidate_u = np.empty([n_nodes, 2], dtype=np.float32)
        candidate_v = np.empty([n_nodes, 2], dtype=np.float32)
        candidate_t = np.empty([n_nodes], dtype=np.float32)
        t_n_labels, t_labels, t_label_probs = None, None, None

        # Concatenate the data from all the nodes at this depth
        c_len_x = np.empty([n_nodes, n_images], dtype=np.int32)
        c_skip_mask = np.empty([n_nodes], dtype=np.bool)
        max_pixels = 0
        for i in range(n_nodes):
            node = queue[i]
            c_len_x[i][:] = node['xl']
            max_node_pixels = np.amax(node['xl'])
            if max_node_pixels > max_pixels:
                max_pixels = max_node_pixels
            c_skip_mask[i] = node['depth'] >= MAX_DEPTH or max_pixels <= 1

        c_all_x = np.zeros([n_nodes, n_images, max_pixels, 2], dtype=np.int32)
        for i in range(n_nodes):
            node = queue[i]
            for j in range(n_images):
                n_pixels = c_len_x[i][j]
                c_all_x[i, j, 0:n_pixels, 0:2] = node['x'][j]

        for epoch in range(1, N_EPOCHS + 1):
            if epoch == 1 or \
               epoch % DISPLAY_STEP == 0 or \
               epoch == N_EPOCHS:
                hours, minutes, seconds = elapsed(begin)
                print('(%02d:%02d:%02d) \tEpoch %dx%d (%d) ' % \
                      (hours, minutes, seconds, \
                       epoch, COMBO_SIZE, epoch * COMBO_SIZE))

            # Run session to find label histograms and candidate gains
            (c_gains, c_u, c_v, c_t) = (None, None, None, None)

            params = { all_x: c_all_x, \
                       len_x: c_len_x, \
                       nodes_size: n_nodes, \
                       skip_mask: c_skip_mask }
            if epoch == 1:
                tensors = (best_gains, best_u, best_v, best_t, \
                           all_n_labels, all_x_labels, all_x_label_probs)
                c_gains, c_u, c_v, c_t, t_n_labels, t_labels, t_label_probs = \
                    session.run(tensors, feed_dict=params)
            else:
                tensors = (best_gains, best_u, best_v, best_t)
                c_gains, c_u, c_v, c_t = \
                    session.run(tensors, feed_dict=params)

            # See what the best-performing u,v,t combination was for each node
            for n in range(n_nodes):
                if c_gains[n] > candidate_G[n]:
                    candidate_G[n] = c_gains[n]
                    candidate_u[n] = c_u[n]
                    candidate_v[n] = c_v[n]
                    candidate_t[n] = thresholds[c_t[n]]

                    if DISPLAY_VERBOSE_PROGRESS:
                        print('\t\tNode \'%s\'' % queue[n]['name'])
                        print('\t\tG = %f' % (candidate_G[n] / n_images))
                        print('\t\tu = ' + str(candidate_u[n]))
                        print('\t\tv = ' + str(candidate_v[n]))
                        print('\t\tt = ' + str(candidate_t[n]))

        # Collect l/r pixels for the best u,v,t combinations for each node
        hours, minutes, seconds = elapsed(begin)
        print('(%02d:%02d:%02d) Collecting pixels...' % \
              (hours, minutes, seconds))

        t_meta_indices, t_lindices, t_rindices = session.run( \
            (all_meta_indices, all_lindices, all_rindices), \
            feed_dict={ u: candidate_u, \
                        v: candidate_v, \
                        t: candidate_t, \
                        all_x: c_all_x, \
                        len_x: c_len_x, \
                        nodes_size: n_nodes })

        # Extract l/r pixels from the previously collected arrays
        lcoords = [[None for i in range(n_images)] for i in range(n_nodes)]
        rcoords = [[None for i in range(n_images)] for i in range(n_nodes)]
        nlcoords = np.empty([n_nodes, n_images], dtype=np.int32)
        nrcoords = np.empty([n_nodes, n_images], dtype=np.int32)
        maxlcoords = np.zeros([n_nodes], dtype=np.int32)
        maxrcoords = np.zeros([n_nodes], dtype=np.int32)

        idx = 0
        lindex_base = 0
        rindex_base = 0
        for i in range(n_images):
            for n in range(n_nodes):
                meta_indices = t_meta_indices[idx]
                idx += 1

                lend = lindex_base + meta_indices[0]
                rend = rindex_base + meta_indices[1]

                lindices = t_lindices[lindex_base:lend]
                rindices = t_rindices[rindex_base:rend]

                lcoords[n][i] = c_all_x[n][i][lindices]
                rcoords[n][i] = c_all_x[n][i][rindices]

                nlcoords[n][i] = meta_indices[0]
                nrcoords[n][i] = meta_indices[1]

                if meta_indices[0] > maxlcoords[n]:
                    maxlcoords[n] = meta_indices[0]
                if meta_indices[1] > maxrcoords[n]:
                    maxrcoords[n] = meta_indices[1]

                lindex_base = lend
                rindex_base = rend

        # If gain hasn't increased for any images, there's no use in going
        # down this branch further.
        # TODO: Maybe this should be a threshold rather than just zero.
        for n in range(n_nodes):
            node = queue.pop(0)
            depth = node['depth']
            if depth < MAX_DEPTH and \
               candidate_G[n] > 0.0 and \
               maxlcoords[n] > 0 and \
               maxrcoords[n] > 0:
                # Store the trained u,v,t parameters
                node['u'] = candidate_u[n]
                node['v'] = candidate_v[n]
                node['t'] = candidate_t[n]

                # Add left/right nodes to the queue
                node['l'] = { 'name': node['name'] + 'l', \
                              'depth': depth + 1, \
                              'x': lcoords[n], \
                              'xl': nlcoords[n] }
                node['r'] = { 'name': node['name'] + 'r', \
                              'depth': depth + 1, \
                              'x': rcoords[n], \
                              'xl': nrcoords[n] }
                queue.extend([node['l'], node['r']])
            else:
                # This is a leaf node, store the label probabilities
                if DISPLAY_VERBOSE_PROGRESS:
                    excuse = None
                    if depth >= MAX_DEPTH:
                        excuse = 'Maximum depth (%d) reached' % (MAX_DEPTH)
                    else:
                        excuse = 'No gain increase'

                    print('\t(%s) Leaf node (%s):' % (node['name'], excuse))

                # Collect label histograms
                n_labels = 0
                label_probs = {}
                idx = n * n_images
                label_base = 0 if idx == 0 else t_n_labels[idx - 1]
                for i in range(n_images):
                    label_end = t_n_labels[idx]
                    for l in range(label_base, label_end):
                        key = t_labels[l]
                        if key not in label_probs:
                            label_probs[key] = t_label_probs[l]
                            n_labels += 1
                        else:
                            label_probs[key] += t_label_probs[l]

                for key, value in label_probs.items():
                    label_probs[key] = value / float(n_labels)
                    if DISPLAY_VERBOSE_PROGRESS:
                        print('\t\t%8d - %0.3f' % (key, label_probs[key]))

                node['label_probs'] = label_probs

            # Remove unneeded data from the node
            del node['name']
            del node['depth']
            del node['x']
            del node['xl']

        # Possibly checkpoint
        now = time.monotonic()
        if exit_after_checkpoint.triggered or \
           now - checkpoint_time > CHECKPOINT_TIME:
            hours, minutes, seconds = elapsed(begin)
            print('(%02d:%02d:%02d) Writing checkpoint...' % \
                  (hours, minutes, seconds))
            write_checkpoint(root, queue)
            checkpoint_time = now
            if exit_after_checkpoint.triggered is True:
                sys.exit()

    coord.request_stop()

    # Save tree
    hours, minutes, seconds = elapsed(begin)
    print('(%02d:%02d:%02d) Writing tree to \'tree.pkl\'' % \
          (hours, minutes, seconds))
    with open('tree.pkl', 'wb') as f:
        pickle.dump(root, f)

    hours, minutes, seconds = elapsed(begin)
    print('(%02d:%02d:%02d) Complete' % (hours, minutes, seconds))

    coord.join(threads)
