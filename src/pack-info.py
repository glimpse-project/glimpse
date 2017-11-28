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

import argparse
from py import pack
import copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_pack", nargs=1, help="Input Pack file")
    parser.add_argument("-o", "--output", nargs=1, help="Output Pack file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose debug information")

    args = parser.parse_args()

    in_filename = args.in_pack[0]
    in_pack = pack.PackReader(in_filename)

    out = None
    if args.output:
        out = pack.PackWriter(args.output[0])
        out.properties = in_pack.copy_properties()
        out.write_header()

    print("version = %u.%u" % (in_pack.major, in_pack.minor))

    print("%u sections:" % len(in_pack.section_names))
    for section_name in in_pack.section_names:
        print("  %s" % section_name)

    print("%u properties:" % len(in_pack.properties))
    for key in in_pack.properties:
        print("  %s = %s" % (key, str(in_pack.properties[key])))

    for frame in in_pack:
        if out:
            new_frame = out.new_frame()

        print("Frame:")

        for key in frame.properties:
            print("  %s = %s" % (key, str(frame.properties[key])))

        if out:
            new_frame.properties = copy.deepcopy(frame.properties)

        print("  sections:")
        for section_name in in_pack.section_names:
            section = frame[section_name]
            print("    %s:" % section.name)
            #data = section.get_decompressed()
            #print("      %s" % str(data))

            compressed = section.get_compressed()

            if out:
                new_section = new_frame[section_name]
                print("    copying across compressed section %s with len %u" % (section_name, len(compressed)))
                new_section.set_compressed(section.get_compressed())

        if out:
            out.append_frame(new_frame)


if __name__ == "__main__":
    main()

