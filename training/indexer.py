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
#
# This script first builds an internal index of all available rendered frames
# in a given directory and can then split that into multiple sets with no
# overlap so for example we could create three sets of 300k images out of a
# large set of 1 million images for training three separate decision trees.


import os
import argparse
import textwrap
import random
import json

full_index = []

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent("""\
        This indexer will firstly write a data/index.full file for all the frames
        it finds under the given data directory.
        
        For each -i <name> <N> argument given it will create a
        data/index.<name> file with <N> randomly sampled frames taken from the
        full index. Each index is created in the order specified and no frames
        will be added to more than one of these name index files.

        Importantly the sampling is pseudo random and reproducible for a given
        directory of data.
        """))
parser.add_argument("data", nargs=1, help="Data Directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose debug information")
parser.add_argument("-s", "--seed", help="Seed for random sampling")
parser.add_argument("-i", "--index", action="append", nargs=2, metavar=('name','N'), help="Request a named index with N frames")

args = parser.parse_args()

random.seed(0xf11bb1e)
if args.seed:
    random.seed(int(args.seed))

data_dir = args.data[0]

for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.startswith("Image") and filename.endswith(".json"):
            frame_name = filename[5:-5]
            (mocap_path, section) = os.path.split(root)
            (top_path,mocap) = os.path.split(mocap_path)

            full_index.append("/%s/%s/Image%s\n" % (mocap, section, frame_name))

            if args.verbose:
                print("mocap = %s, section = %s, frame = %s" % (mocap, section, frame_name))


n_frames = len(full_index)

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

with open(os.path.join(data_dir, "index.full"), 'w+') as fp:
    for frame in full_index:
        fp.write(frame)
print("index.full: %u frames" % n_frames)
