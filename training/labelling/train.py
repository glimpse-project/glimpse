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
from collections import deque
from tensorflow.python.client import timeline

# Seed to use for random number generation
RANDOM_SEED = 1
# Image dimensions
WIDTH = 540
HEIGHT = 960
# Pixels per meter
PPM = 579.0
# Number of images to pre-load in the queue
QUEUE_BUFFER = 200
# Number of threads to use when pre-loading queue
QUEUE_THREADS = multiprocessing.cpu_count()
# Limit the number of images in the training set
DATA_LIMIT = 0
# Number of epochs to train per node
N_EPOCHS = 1
# Maximum number of nodes to train at once
MAX_NODES = 2048
# Maximum number of u,v pairs to test per epoch. The paper specifies testing
# 2000 candidate u,v pairs (the equivalent number is N_EPOCHS * COMBO_SIZE)
COMBO_SIZE = 2000
# How frequently to display epoch progress
DISPLAY_STEP = 1
# How long to let elapse before creating a checkpoint (in seconds)
CHECKPOINT_TIME = 1800
# Whether to display verbose progress
DISPLAY_VERBOSE_PROGRESS = False
# Depth to train tree to (paper specifies 20)
MAX_DEPTH = 20
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
# Whether to do profiling
PROFILE = False
# If profiling is enabled, whether to write out a Chrome trace
CHROME_PROFILE = False
# Whether to enable XLA JIT for graph optimisation
XLA_JIT = True

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

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
    extents = [HEIGHT - 1, WIDTH - 1]
    n_pixels = tf.size(pixels)
    pixels = tf.cast(pixels, tf.float32)

    u = tf.tile([u], [n_pixels, 1]) / pixels
    v = tf.tile([v], [n_pixels, 1]) / pixels

    # Note, we clip the coordinates here. The paper says that any coordinate
    # that references outside of the image should essentially result in the
    # background depth - we can assure this by processing the input images.
    x = tf.cast(x, tf.float32)
    uindices = tf.clip_by_value(tf.cast(tf.round(x + u), tf.int32), 0, extents)
    vindices = tf.clip_by_value(tf.cast(tf.round(x + v), tf.int32), 0, extents)

    return uindices, vindices

def shannon_entropy(values):
    y, idx, count = tf.unique_with_counts(values)
    count = tf.cast(count, tf.float32)
    ncount = count / tf.reduce_sum(count)
    return -tf.reduce_sum(ncount * (tf.log(ncount) / tf.log(2.0))), y, ncount

def computeDepthPixels(depth_image, depth_pixels, u, v, x):
    with jit_scope(compile_ops=XLA_JIT):
        # Gather the candidate pixels from the depth image and apply
        # equation (1) from 3.2 of the paper
        uindices, vindices = get_offset_indices(depth_image, depth_pixels, x, u, v)
        return tf.squeeze( \
            tf.gather_nd(depth_image, uindices) - \
            tf.gather_nd(depth_image, vindices), [1])

def computeLabelHistogramAndEntropy(label_pixels):
    with jit_scope(compile_ops=XLA_JIT):
        # Compute the shannon entropy for storage of the labels of the candidate
        # depth pixels.
        hq, x_labels, x_label_prob = shannon_entropy(label_pixels)
        q = tf.cast(tf.size(label_pixels), tf.float32)

        return hq, q, x_labels, x_label_prob

def splitOnThreshold(splitee, splitter, t):
    partitions = tf.cast( \
        tf.clip_by_value(splitter - (t - 1.0), 0.0, 1.0), tf.int32)
    return tf.dynamic_partition(splitee, partitions, 2)

def computeGain(llabel_pixels, rlabel_pixels, hq, q):
    with jit_scope(compile_ops=XLA_JIT):
        # Compute gain (see equation (6) from 3.3 of the paper)
        hql, l_labels, l_label_prob = shannon_entropy(llabel_pixels)
        ql = tf.cast(tf.shape(llabel_pixels)[0], tf.float32)

        hqr, r_labels, r_label_prob = shannon_entropy(rlabel_pixels)
        qr = tf.cast(tf.shape(rlabel_pixels)[0], tf.float32)

        return hq - ((ql / q * hql) + (qr / q * hqr))

def computeThresholdGains(depth_image, depth_pixels, u, v, x, label_pixels, hq, q):
    # Compute the set of pixels that will be used for partitioning
    fdepth_pixels = computeDepthPixels(depth_image, depth_pixels, u, v, x)

    def computeThresholdGain(i, t, gains):
        l_pixels, r_pixels = splitOnThreshold(label_pixels, fdepth_pixels, t)
        G = tf.cond(tf.size(l_pixels) < 1, \
                    lambda: 0.0, \
                    lambda: tf.cond(tf.size(r_pixels) < 1, \
                                    lambda: 0.0, \
                                    lambda: computeGain(l_pixels, r_pixels, hq, q)), \
                    name='computeGain')
        return i + 1, t + T_INC, gains.write(i, G)

    gains = tf.TensorArray(tf.float32, N_T, name='threshold_gains')
    _i, _t, gains = tf.while_loop( \
        lambda i, t, gains: i < N_T, \
        computeThresholdGain, \
        [0, MIN_T, gains], \
        parallel_iterations=N_T, \
        back_prop=False, name='threshold_loop')

    return gains.stack()

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
label_image = tf.image.decode_png(label_value, channels=1)

depth_image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, dtypes=(tf.float32), \
                                 name='depth_queue')
label_image_queue = tf.FIFOQueue(capacity=QUEUE_BUFFER, dtypes=(tf.uint8), \
                                 name='label_queue')
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
all_u = tf.random_uniform([nodes_size, COMBO_SIZE, 2], MIN_UV, MAX_UV, \
                          name='all_u')
all_v = tf.random_uniform([nodes_size, COMBO_SIZE, 2], MIN_UV, MAX_UV, \
                          name='all_v')

# Pixel coordinates to test
all_x = tf.placeholder(tf.int32, shape=[None, 2], name='pixels')
x_index = tf.placeholder(tf.int32, shape=[None, n_images, 2], name='pixel_indices')

# Which nodes to test
skip_mask = tf.placeholder(tf.bool, shape=[None], name='skip_mask')

def accumulate_gain(i, acc_gain, all_n_labels, all_x_labels, all_x_label_probs):
    # Read in depth and label image
    depth_image = depth_image_queue.dequeue()
    label_image = label_image_queue.dequeue()
    depth_image.set_shape([HEIGHT, WIDTH])
    label_image.set_shape([HEIGHT, WIDTH])

    base = i * nodes_size

    def test_node(n, node_gains, all_x_labels, all_x_label_probs, all_n_labels):
        # Slice out the appropriate values for the image
        x_start = x_index[n][i][0]
        x_size = x_index[n][i][1]
        x = tf.slice(all_x, [x_start, 0], [x_size, -1])

        depth_pixels = tf.gather_nd(depth_image, x, name='depth_pixels')
        label_pixels = tf.squeeze(tf.gather_nd( \
            label_image, x, name='label_pixels'), [1])

        # Compute histogram shannon entropy
        hq, q, x_labels, x_label_prob = \
            computeLabelHistogramAndEntropy(label_pixels)

        # Keep track of the gains for this node, but short-circuit if there are
        # no label pixels or we're not testing this node
        def add_gain():
            # Test u,v pairs against a range of thresholds
            def test_uv(c, all_gain):
                gains = computeThresholdGains(depth_image, depth_pixels, \
                                              all_u[n][c], all_v[n][c], \
                                              x, label_pixels, hq, q)

                return c + 1, all_gain.write(c, gains)

            all_gain = tf.TensorArray(tf.float32, COMBO_SIZE, \
                                      name='combo_gains')
            _c, all_gain = \
                tf.while_loop( \
                    lambda c, all_gain: c < COMBO_SIZE, \
                    test_uv, [0, all_gain], \
                    parallel_iterations=COMBO_SIZE, \
                    back_prop=False, name='combo_loop')

            return node_gains.write(n, all_gain.stack())

        def skip_gain():
            return node_gains.write(n, tf.zeros([COMBO_SIZE, N_T]))

        node_gains = tf.cond(skip_mask[n], \
            lambda: skip_gain(),
            lambda: tf.cond(tf.size(x_labels) < 2, \
                            lambda: skip_gain(),
                            lambda: add_gain()), name='collect_gain')

        # Collect label histogram data
        idx = base + n

        # xlabels is [?], int32
        #   (Unique labels for pixels in x)
        all_x_labels = all_x_labels.write(idx, x_labels)

        # x_label_prob is [len(xlabels)], float32
        #   (Distribution of each label in xlabels)
        all_x_label_probs = all_x_label_probs.write(idx, x_label_prob)

        # Store the index of the label histogram
        all_n_labels = all_n_labels.write(idx, tf.size(x_labels))

        return n+1, node_gains, all_x_labels, all_x_label_probs, all_n_labels

    node_gains = tf.TensorArray(tf.float32, nodes_size, name='node_gains')
    _n, node_gains, all_x_labels, all_x_label_probs, all_n_labels = \
        tf.while_loop( \
            lambda n, _ng, _xl, _xlp, _nl: n < nodes_size,
            test_node,
            [0, node_gains, all_x_labels, all_x_label_probs, all_n_labels], \
            parallel_iterations=MAX_NODES, \
            back_prop=False, name='node_loop')

    # Accumulate gain
    acc_gain += node_gains.stack()

    return i + 1, acc_gain, all_n_labels, all_x_labels, all_x_label_probs

# Run n_images iterations over COMBO_SIZE u,v pairs
acc_gain = \
    tf.zeros([nodes_size, COMBO_SIZE, N_T], dtype=tf.float32, name='acc_gain')
all_n_labels = \
    tf.TensorArray(tf.int32, n_images * nodes_size, name='all_n_labels')
all_x_labels = \
    tf.TensorArray(tf.uint8, 0, dynamic_size=True, name='all_x_labels')
all_x_label_probs = \
    tf.TensorArray(tf.float32, 0, dynamic_size=True, name='all_x_label_probs')

_i, acc_gain, all_n_labels, all_x_labels, all_x_label_probs = tf.while_loop( \
    lambda i, a, b, c, d: i < n_images, \
    accumulate_gain, \
    [0, acc_gain, all_n_labels, all_x_labels, all_x_label_probs], \
    parallel_iterations = 1, \
    back_prop=False, name='image_loop')

all_x_labels = all_x_labels.concat()
all_x_label_probs = all_x_label_probs.concat()

# Scan over all_n_labels to make the indices absolute
all_n_labels = tf.scan(lambda a, x: a + x, all_n_labels.stack(), \
                       back_prop=False, name='label_acc_scan')

# Find the best gain and the best combination index
flat_gain = tf.reshape(acc_gain, [nodes_size, COMBO_SIZE * N_T], \
                       name='flat_gain')
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
                      tf.TensorShape([None])], \
    back_prop=False, name='best_combo_loop')

# Construct graph for retrieving lr pixel coordinates
# For retrieving the coordinates of a particular u,v,t combination
u = tf.placeholder(tf.float32, shape=[None, 2], name='u')
v = tf.placeholder(tf.float32, shape=[None, 2], name='v')
t = tf.placeholder(tf.float32, shape=[None], name='t')

def collect_indices(i, all_meta_indices, all_lcoords, all_rcoords):
    # Dequeue the depth image
    depth_image = depth_image_queue.dequeue()
    depth_image.set_shape([HEIGHT, WIDTH])

    def collect_node_indices(n, all_meta_indices, all_lcoords, all_rcoords):
        # Slice out the appropriate values for the image
        x_start = x_index[n][i][0]
        x_size = x_index[n][i][1]
        x = tf.slice(all_x, [x_start, 0], [x_size, -1])

        def collectDepthPixels():
            # Compute the depth pixels that we'll need to compare against the threshold
            depth_pixels = tf.gather_nd(depth_image, x)
            fdepth_pixels = computeDepthPixels(depth_image, depth_pixels, u[n], v[n], x)

            # Get the left/right pixels
            lcoords, rcoords = splitOnThreshold(x, fdepth_pixels, t[n])

            # Work out index ranges
            lsize = tf.shape(lcoords)[0]
            rsize = tf.shape(rcoords)[0]
            meta_index = [lsize, rsize]

            return lcoords, rcoords, meta_index

        def collectZeroPixels():
            return tf.zeros([0, 2], dtype=tf.int32), \
                   tf.zeros([0, 2], dtype=tf.int32), [0, 0]

        lcoords, rcoords, meta_index = \
            tf.cond(tf.shape(x)[0] < 1, \
                    lambda: collectZeroPixels(), \
                    lambda: collectDepthPixels(), name='collect_depth_pixels')

        # Add pixels to pixel lists
        all_meta_indices = tf.concat([all_meta_indices, [meta_index]], axis=0)
        all_lcoords = tf.concat([all_lcoords, lcoords], axis=0)
        all_rcoords = tf.concat([all_rcoords, rcoords], axis=0)

        return n+1, all_meta_indices, all_lcoords, all_rcoords

    _n, all_meta_indices, all_lcoords, all_rcoords = tf.while_loop( \
        lambda n, _ami, _al, _ar: n < nodes_size, \
        collect_node_indices, \
        [0, all_meta_indices, all_lcoords, all_rcoords],
        shape_invariants=[tf.TensorShape([]), \
                          tf.TensorShape([None, 2]), \
                          tf.TensorShape([None, 2]), \
                          tf.TensorShape([None, 2])], \
        back_prop=False, name='lr_collection_node_loop')

    return i + 1, all_meta_indices, all_lcoords, all_rcoords

all_meta_indices = tf.zeros([0, 2], dtype=tf.int32)
all_lcoords = tf.zeros([0, 2], dtype=tf.int32)
all_rcoords = tf.zeros([0, 2], dtype=tf.int32)

_i, all_meta_indices, all_lcoords, all_rcoords = tf.while_loop( \
    lambda i, a, b, c: i < n_images, \
    collect_indices, \
    [0, all_meta_indices, all_lcoords, all_rcoords], \
    shape_invariants=[tf.TensorShape([]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None, 2]), \
                      tf.TensorShape([None, 2])], \
    back_prop=False, name='lr_collection_image_loop')

# Initialise and run the session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
if XLA_JIT:
    config.graph_options.optimizer_options.global_jit_level = \
        tf.OptimizerOptions.ON_1
session = tf.Session(config=config)

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
                 'depth': 1,
                 'x': np.reshape(initial_coords, (n_images, N_SAMP, 2)) }
        queue = [root]

    # Initialise the output tree
    thresholds = np.linspace(MIN_T, MAX_T, N_T)

    # Start timing
    begin = time.monotonic()
    checkpoint_time = begin

    # Setup signal handler
    exit_after_checkpoint = BreakHandler()

    # Setup session options/metadata
    options = None
    profile_no = 0
    summary_writer = None
    run_metadata = tf.RunMetadata()
    profile_train_dir = os.path.join('profile', 'train')
    profile_collect_dir = os.path.join('profile', 'train')
    if PROFILE:
        if CHROME_PROFILE:
            os.makedirs(profile_train_dir, exist_ok=True)
            os.makedirs(profile_collect_dir, exist_ok=True)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        summary_dir = os.path.join('profile', 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(summary_dir, session.graph)
    else:
        options = tf.RunOptions()

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
        c_all_x = deque([])
        c_x_index = np.empty([n_nodes, n_images, 2], dtype=np.int32)
        c_skip_mask = np.empty([n_nodes], dtype=np.bool)
        max_pixels = 0
        idx_start = 0
        for n in range(n_nodes):
            node = queue[n]

            max_node_pixels = 0
            pixels = node['x']
            for i in range(n_images):
                n_pixels = len(pixels[i])
                if n_pixels > max_node_pixels:
                    max_node_pixels = n_pixels

                c_x_index[n][i] = [idx_start, n_pixels]
                c_all_x.extend(pixels[i])
                idx_start += n_pixels

            if max_node_pixels > max_pixels:
                max_pixels = max_node_pixels
            c_skip_mask[n] = node['depth'] >= MAX_DEPTH or max_node_pixels <= 1
        c_all_x = np.array(c_all_x)

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
                       x_index: c_x_index, \
                       nodes_size: n_nodes, \
                       skip_mask: c_skip_mask }
            if epoch == 1:
                tensors = (best_gains, best_u, best_v, best_t, \
                           all_n_labels, all_x_labels, all_x_label_probs)
                c_gains, c_u, c_v, c_t, t_n_labels, t_labels, t_label_probs = \
                    session.run(tensors, feed_dict=params, options=options, \
                                run_metadata=run_metadata)
            else:
                tensors = (best_gains, best_u, best_v, best_t)
                c_gains, c_u, c_v, c_t = \
                    session.run(tensors, feed_dict=params, options=options, \
                                run_metadata=run_metadata)

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

            if PROFILE:
                hours, minutes, seconds = elapsed(begin)
                print('(%02d:%02d:%02d) Writing training profile data...' % \
                      (hours, minutes, seconds))

                if CHROME_PROFILE:
                    run_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = run_timeline.generate_chrome_trace_format()
                    chrome_filename = \
                        os.path.join(profile_train_dir, \
                                     '%02d_%02d_profile.json' % \
                                     (profile_no, epoch))
                    with open(chrome_filename, 'w') as f:
                        f.write(chrome_trace)

                summary_writer.add_run_metadata(run_metadata, \
                                                '%02d_%02d' % (profile_no, epoch))
                summary_writer.flush()

        # Collect l/r pixels for the best u,v,t combinations for each node
        hours, minutes, seconds = elapsed(begin)
        print('(%02d:%02d:%02d) Collecting pixels...' % \
              (hours, minutes, seconds))

        t_meta_indices, t_lcoords, t_rcoords = session.run( \
            (all_meta_indices, all_lcoords, all_rcoords), \
            feed_dict={ u: candidate_u, \
                        v: candidate_v, \
                        t: candidate_t, \
                        all_x: c_all_x, \
                        x_index: c_x_index, \
                        nodes_size: n_nodes }, \
            options=options, run_metadata=run_metadata)

        if PROFILE:
            hours, minutes, seconds = elapsed(begin)
            print('(%02d:%02d:%02d) Writing collection profile data...' % \
                  (hours, minutes, seconds))

            if CHROME_PROFILE:
                run_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = run_timeline.generate_chrome_trace_format()
                with open('%02dc_profile.json' % (profile_no), 'w') as f:
                    f.write(chrome_trace)

            summary_writer.add_run_metadata(run_metadata, '%02dc' % (profile_no))
            summary_writer.flush()

            profile_no += 1

        # Extract l/r pixels from the previously collected arrays
        lcoords = [[None for i in range(n_images)] for i in range(n_nodes)]
        rcoords = [[None for i in range(n_images)] for i in range(n_nodes)]
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

                lcoords[n][i] = t_lcoords[lindex_base:lend]
                rcoords[n][i] = t_rcoords[rindex_base:rend]

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
                              'x': lcoords[n] }
                node['r'] = { 'name': node['name'] + 'r', \
                              'depth': depth + 1, \
                              'x': rcoords[n] }
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
                for i in range(n, n + (n_images * n_nodes), n_nodes):
                    label_base = 0 if i == 0 else t_n_labels[i - 1]
                    label_end = t_n_labels[i]
                    if label_end > label_base:
                        n_labels += 1
                        for l in range(label_base, label_end):
                            key = t_labels[l]
                            if key not in label_probs:
                                label_probs[key] = t_label_probs[l]
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

        # Possibly checkpoint
        now = time.monotonic()
        if exit_after_checkpoint.triggered or \
           (CHECKPOINT_TIME >= 0 and now - checkpoint_time > CHECKPOINT_TIME):
            hours, minutes, seconds = elapsed(begin)
            print('(%02d:%02d:%02d) Writing checkpoint...' % \
                  (hours, minutes, seconds))
            write_checkpoint(root, queue)
            checkpoint_time = now
            if exit_after_checkpoint.triggered is True:
                sys.exit()

    coord.request_stop()

    if PROFILE:
        summary_writer.close()

    # Save tree
    hours, minutes, seconds = elapsed(begin)
    print('(%02d:%02d:%02d) Writing tree to \'tree.pkl\'' % \
          (hours, minutes, seconds))
    with open('tree.pkl', 'wb') as f:
        pickle.dump(root, f)

    hours, minutes, seconds = elapsed(begin)
    print('(%02d:%02d:%02d) Complete' % (hours, minutes, seconds))

    coord.join(threads)
