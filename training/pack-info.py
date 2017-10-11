#!/usr/bin/env python3

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

