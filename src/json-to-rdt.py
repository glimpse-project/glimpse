#!/usr/bin/env python3
#
# Copyright (c) 2017 Glimp IP Ltd
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
# This script takes a JSON randomized decision tree description and packs it
# into a binary representation that's understood by our inferrence code.

import os
import argparse
import textwrap
import json
import struct

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent("""\
        This tool converts the JSON representation of the randomised decision trees
        output by the train_rdt tool to a binary representation which can be
        convenient for fast loading of trees and more compact representation when
        compressed.

        Note: A packed RDT file only needs to contain the minimum information for
              efficient runtime inference so the conversion is lossy.

        This reads a JSON description of a randomized decision tree with the
        following schema:

          {
            "n_labels": 34,
            "vertical_fov": 50.4,
            "depth": 20,
            "root": {
              "t": 0.5,            // threshold
              "u": [  3, -0.5 ],   // U vector
              "v": [ -2, 10.1 ],   // V vector
      
              // left node...
              "l": {
                // children directly nested...
                "t": 0.2,          // child's threshold
                "u": [  1, -5.2 ], // child's U vector
                "v": [ -7, -0.1 ], // child's V vector
                "l": { ... },      // child's left node
                "r": { ... },      // child's right node
              },
      
              // right node is an example leaf node with <n_labels> label
              // probabilities...
              "r": {
                "p": [0, 0, 0, 0.2, 0, 0.2, 0, 0.6, 0, 0 ... ],
              }
            }
          }

        """))
parser.add_argument("json_file", nargs=1, help="Input JSON Decision Tree")
parser.add_argument("rdt_file", nargs=1, help="Output RDT Decision Tree")

args = parser.parse_args()

with open(args.json_file[0], 'r') as json_fp:
    tree = json.load(json_fp)

    depth = int(tree['depth'])
    n_labels = int(tree['n_labels'])
    bg_label = int(tree['bg_label'])
    fov = float(tree['vertical_fov'])

    with open(args.rdt_file[0], 'wb+') as fp:
        # 11 byte v4 RDTHeader
        #
        # Note: the leading '<' specifies little endian and also ensures the
        # data is packed without alignment padding between the first 6 bytes
        # and the float fov)
        fp.write(struct.pack('<3s4Bf',
                             "RDT".encode('ascii'),
                             4, # Version
                             depth,
                             n_labels,
                             bg_label,
                             fov
                             ))

        # NB: the .uv member of the C Node struct is 16 byte aligned resulting
        # in 4 bytes padding at the end of the struct.
        sizeof_v4_Node = 32
        n_nodes = pow(2, depth) - 1

        def count_probability_arrays(node):
            if 'p' in node:
                return 1
            return (count_probability_arrays(node['l']) +
                    count_probability_arrays(node['r']))

        def pack_probability_arrays(buf, node, next_idx=0):
            if 'p' in node:
                # NB: the label_pr_idx should be a base-one index since index zero
                # is reserved to indicate that a node is not a leaf node
                node['label_pr_idx'] = next_idx + 1

                off = next_idx * 4 * n_labels
                for i in range(n_labels):
                    struct.pack_into('f', buf, off, node['p'][i])
                    off += 4
                next_idx += 1
            else:
                next_idx = pack_probability_arrays(buf, node['l'], next_idx)
                next_idx = pack_probability_arrays(buf, node['r'], next_idx)
            return next_idx

        n_pr_arrays = count_probability_arrays(tree['root'])
        pr_array = bytearray(n_pr_arrays * n_labels * 4)
        pack_probability_arrays(pr_array, tree['root'])

        def pack_node(buf, node, idx):
            off = sizeof_v4_Node * idx
            if 't' in node:
                struct.pack_into('ff', buf, off,    node['u'][0], node['u'][1])
                struct.pack_into('ff', buf, off+8,  node['v'][0], node['v'][1])
                struct.pack_into('f',  buf, off+16, node['t'])

                # With a complete tree in breadth-first order (with left
                # followed by right) and root at index zero then given an
                # index for any node we can calculate the index of the
                # left child as index * 2 + 1 and the right as index * 2 + 2
                pack_node(buf, node['l'], idx * 2 + 1)
                pack_node(buf, node['r'], idx * 2 + 2)
            else:
                struct.pack_into('I',  buf, off+20, node['label_pr_idx'])
        
        node_array = bytearray(sizeof_v4_Node * n_nodes);
        pack_node(node_array, node=tree['root'], idx=0)
        fp.write(node_array)

        fp.write(pr_array)

