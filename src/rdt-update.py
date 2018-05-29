#!/usr/bin/env python3
#
# Copyright (c) 2018 Glimp IP Ltd
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

import os
import argparse
import textwrap
import json
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent("""\
        This tool updates pre-existing .json decision trees to conform to newer
        schemas.
        """))
parser.add_argument("json_file", nargs=1, help="JSON decision tree to update")
parser.add_argument("-p", "--pretty", action='store_true', help='Write indented JSON')

args = parser.parse_args()

default_labels = [
    {
        "name": "head left",
        "opposite": "head right"
    },
    {
        "name": "head right",
        "opposite": "head left"
    },
    {
        "name": "head top left",
        "opposite": "head top right"
    },
    {
        "name": "head top right",
        "opposite": "head top left"
    },
    {
        "name": "neck",
    },
    {
        "name": "clavicle left",
        "opposite": "clavicle right"
    },
    {
        "name": "clavicle right",
        "opposite": "clavicle left"
    },
    {
        "name": "shoulder left",
        "opposite": "shoulder right"
    },
    {
        "name": "upper-arm left",
        "opposite": "upper-arm right"
    },
    {
        "name": "shoulder right",
        "opposite": "shoulder left"
    },
    {
        "name": "upper-arm right",
        "opposite": "upper-arm left"
    },
    {
        "name": "elbow left",
        "opposite": "elbow right"
    },
    {
        "name": "forearm left",
        "opposite": "forearm right"
    },
    {
        "name": "elbow right",
        "opposite": "elbow left"
    },
    {
        "name": "forearm right",
        "opposite": "forearm left"
    },
    {
        "name": "left wrist",
        "opposite": "right wrist"
    },
    {
        "name": "left hand",
        "opposite": "right hand"
    },
    {
        "name": "right wrist",
        "opposite": "left wrist"
    },
    {
        "name": "right hand",
        "opposite": "left hand"
    },
    {
        "name": "left hip",
        "opposite": "right hip"
    },
    {
        "name": "left thigh",
        "opposite": "right thigh"
    },
    {
        "name": "right hip",
        "opposite": "left hip"
    },
    {
        "name": "right thigh",
        "opposite": "left thigh"
    },
    {
        "name": "left knee",
        "opposite": "right knee"
    },
    {
        "name": "left shin",
        "opposite": "right shin"
    },
    {
        "name": "right knee",
        "opposite": "left knee"
    },
    {
        "name": "right shin",
        "opposite": "left shin"
    },
    {
        "name": "left ankle",
        "opposite": "right ankle"
    },
    {
        "name": "left toes",
        "opposite": "right toes"
    },
    {
        "name": "right ankle",
        "opposite": "left ankle"
    },
    {
        "name": "right toes",
        "opposite": "left toes"
    },
    {
        "name": "left waist",
        "opposite": "right waist"
    },
    {
        "name": "right waist",
        "opposite": "left waist"
    },
    {
        "name": "background",
    }
]

tree = None

changed = False

with open(args.json_file[0], 'r') as json_fp:
    tree = json.load(json_fp)

    if 'root' not in tree or 'n_labels' not in tree:
        sys.exit("Not a decision tree")

    if 'labels' not in tree:
        print("Adding labels")
        tree['labels'] = default_labels
        changed = True

if not changed:
    sys.exit("No changes")

if tree != None:
    print("Writting new file")
    with open(args.json_file[0], 'w') as json_fp:
        if args.pretty:
            json.dump(tree, json_fp, indent=4)
        else:
            json.dump(tree, json_fp)
