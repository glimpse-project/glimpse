#!/usr/bin/env python
#
# Copyright (c) 2017 Impossible Labs
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import os
import sys
import subprocess
import argparse


_args = None
_sysroot_dir = None


def run_interactive(args):
    if _args.verbose:
        print("# " + " ".join(map(str, args)), file=sys.stderr)
        returncode = subprocess.call(args)
        print("# return status = " + str(returncode))
        return returncode
    else:
        return subprocess.call(args)


def mkdir(directory):
    if not os.path.exists(directory):
        print("# os.makedirs('" + directory + "')")
        os.makedirs(directory)


def fetch(device_path):

    print("Fetching " + device_path)

    local_dir = os.path.dirname(os.path.join(_sysroot_dir, device_path[1:]))
    mkdir(local_dir)
    run_interactive(["adb", "pull", device_path, local_dir])


def main():
    global _args
    global _sysroot_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("sysroot_directory", nargs="+", help="The directory to download the device sysroot into")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose debug information")

    _args = parser.parse_args()

    _sysroot_dir = os.path.abspath(_args.sysroot_directory[0]);

    fetch("/system/lib")
    fetch("/system/lib64")
    fetch("/system/bin")
    fetch("/system/vendor/lib")
    fetch("/system/vendor/lib64")
    fetch("/vendor/lib")
    fetch("/vendor/lib64")


if __name__ == '__main__':
    main()
