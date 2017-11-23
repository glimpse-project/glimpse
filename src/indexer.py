#!/usr/bin/env python3
#
# Copyright (c) 2017 Kwamecorp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import argparse
import textwrap
import random
import json

full_index = []

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        This script builds an index of all available rendered frames in a given
        directory and can then split that into multiple sub sets with no
        overlap. For example you could index three sets of 300k images out of a
        larger set of 1 million images for training three separate decision
        trees.
        """),
    epilog=textwrap.dedent("""\
        Firstly if no index.full file can be loaded listing all available
        frames then one will be created by traversing all the files under the
        <data> directory looking for frames. --full can be used to override
        which index is loaded here.

        Then if you need to exclude files from the randomly sampled index files
        you can pass options like -e <name1> -e <name2> and the frames listed in
        data/index.<name1> and data/index.<name2> will be loaded as blacklists
        of frames to not sample.
        
        Finally for each -i <name> <N> argument sequence given it will create a
        data/index.<name> file with <N> randomly sampled frames taken from the
        full index. Each index is created in the order specified and no frames
        will be added to more than one of these name index files.

        Importantly the sampling is pseudo random and reproducible for a given
        directory of data. The seed can be explicitly given via --seed=

        Note: Even if the exclusions from passing -e have no effect the act of
        passing -e options can change the random sampling compared to running
        with no exclusion due to how the set difference calculations may affect
        the sorting of the index internally.
        """))
parser.add_argument("data", nargs=1, help="Data Directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose debug information")
parser.add_argument("-s", "--seed", help="Seed for random sampling")
parser.add_argument("-f", "--full", nargs=1, default=['full'], help="An alternative index.<FULL> extension for the full index (default 'full')")
parser.add_argument("-e", "--exclude", action="append", nargs=1, metavar=('NAME'), help="Load index.<NAME> frames to be excluded from sampling")
parser.add_argument("-i", "--index", action="append", nargs=2, metavar=('NAME','N'), help="Create an index.<NAME> file with N frames")

args = parser.parse_args()

random.seed(0xf11bb1e)
if args.seed:
    random.seed(int(args.seed))

data_dir = args.data[0]
full_filename = os.path.join(data_dir, "index.%s" % args.full[0])


# 1. Load the full index
try:
    with open(full_filename, 'r') as fp:
        full_index = fp.readlines()
except FileNotFoundError as e:
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.startswith("Image") and filename.endswith(".json"):
                frame_name = filename[5:-5]
                (mocap_path, section) = os.path.split(root)
                (top_path,mocap) = os.path.split(mocap_path)

                full_index.append("/%s/%s/Image%s\n" % (mocap, section, frame_name))

                if args.verbose:
                    print("mocap = %s, section = %s, frame = %s" %
                            (mocap, section, frame_name))

    with open(full_filename, 'w+') as fp:
        for frame in full_index:
            fp.write(frame)

n_frames = len(full_index)
print("index.%s: %u frames\n" % (args.full[0], n_frames))


# 2. Apply exclusions
if args.exclude:
    exclusions = []
    for (name,) in args.exclude:
        with open(os.path.join(data_dir, 'index.%s' % name), 'r') as fp:
            lines = fp.readlines()
            print("index.%s: loaded %u frames to exclude" % (name, len(lines)))
            exclusions += lines

    full_set = set(full_index)
    exclusion_set = set(exclusions)
    difference = full_set.difference(exclusion_set)
    full_index = list(difference)

    n_frames = len(full_index)
    print("\n%u frames left after applying exclusions" % n_frames)
    print("sorting...")
    full_index.sort()


# 3. Create randomly sampled index files
if args.index:
    names = {}
    total_samples = 0

    for (name, length_str) in args.index:
        total_samples += int(length_str)
        if name in names:
            raise ValueError("each index needs a unique name")
        names[name] = 1

    if total_samples > n_frames:
        raise ValueError("Not enough frames to create requested index files")

    samples = random.sample(range(n_frames), total_samples)

    start = 0
    if args.verbose:
        for (name, length_str) in args.index:
            N = int(length_str)
            subset = samples[start:start+N]
            print("index %s sample indices: %s" % (name, str(subset)))

    print("")
    start = 0
    for (name, length_str) in args.index:
        N = int(length_str)

        with open(os.path.join(data_dir, "index.%s" % name), 'w+') as fp:
            index = []
            subset = samples[start:start+N]
            for i in subset:
                fp.write(full_index[i])
            start += N
            print("index.%s: %u frames" % (name, N))

