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
        This tool prints information about a Glimpse decision trees in a JSON
        format.
        """))
parser.add_argument("json_file", nargs=1, help="Input JSON Decision Tree")
parser.add_argument("-e", "--eval", help='Evaluate exression like "info[\'n_nodes\']" to read single JSON value')

args = parser.parse_args()

def log(*args):
    print(*args)

with open(args.json_file[0], 'r') as json_fp:
    info = json.load(json_fp)
    root = info['root']

    if '_rdt_version_was' in info:
        del info['_rdt_version_was']
    del info['root']
    del info['depth'] # rather read explicitly by traversing tree

    def traverse(info, depth, node):
        if len(info['depths']) < depth + 1:
            dinfo = {
                'n_nodes': 0,
                'n_leaves': 0
            }
            info['depths'].append(dinfo)
        else:
            dinfo = info['depths'][depth]
        if depth + 1 > info['depth']:
            info['depth'] = depth + 1
        info['n_nodes'] += 1
        dinfo['n_nodes'] += 1

        if 'p' in node:
            info['n_leaves'] += 1
            dinfo['n_leaves'] += 1
        if 'l' in node:
            traverse(info, depth + 1, node['l'])
        if 'r' in node:
            traverse(info, depth + 1, node['r'])

    info['depth'] = 0
    info['n_nodes'] = 0
    info['n_leaves'] = 0
    info['depths'] = []
    traverse(info, 0, root)

    info['max_nodes'] = pow(2, info['depth']) - 1

    for d in info['depths']:
        resolved = (d['n_leaves'] / d['n_nodes']) * 100.0
        d['resolved_percent'] = resolved
        d['resolved_bar'] = '*' * int((resolved / 2) + 0.5)

    if args.eval != None:
        val = eval(args.eval)
        log(json.dumps(val, indent=2))
    else:
        log(json.dumps(info, indent=2))
