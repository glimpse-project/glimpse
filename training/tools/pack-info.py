#!/usr/bin/env python3





import argparse
import sys
import struct
import snappy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs=1, help="Pack file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display verbose debug information")

    args = parser.parse_args()

    filename = args.filename[0]
    with open(filename, 'rb') as fp:
        magic = fp.read(5)[:4].decode('ascii')

        if magic != "pack":
            print("%s not a pack file" % filename)
            sys.exit(1)

        (hdr_len,) = struct.unpack('I', fp.read(4))
        print("OK %u" % hdr_len)

        snappy_header = fp.read(hdr_len)
        header = snappy.decompress(snappy_header)

        (major, minor, part0_size, frames_offset, n_properties, n_sections) = struct.unpack('6I', header[:24])

        print("version = %u.%u" % (major, minor))

        if major != 1:
            print("Unsupported version")
            sys.exit(1)

        print("%u sections:" % n_sections)

        for n in range(0, n_sections):
            off = 24 + n * 64
            section_name = header[off:off+64].decode('ascii').split('\0')[0]
            print("  %s" % section_name)

        print("%u properties:" % n_properties)
        def unpack_properties(data):
            off = 0
            properties = {}
            for n in range(0, n_properties):
                prop_name = data[off:off+16].decode('ascii').split('\0')[0]
                (prop_type, byte_len) = struct.unpack('II', data[off+16:off+24])
                if prop_type == 0: # double
                    (val,) = struct.unpack('d', data[off+24:off+32])
                    properties[prop_name] = val
                elif prop_type == 1: # i64
                    (val,) = struct.unpack('q', data[off+24:off+32])
                    properties[prop_name] = val
                elif prop_type == 2: # string
                    val = data[off+24:off+byte_len].decode('utf-8')
                    properties[prop_name] = val
                elif prop_type == 2: # blob
                    val = data[off+24:off+byte_len]
                    properties[prop_name] = val
                else:
                    print("unknown property type")

                off += byte_len

            return properties

        properties = unpack_properties(header[24 + n_sections * 64:])
        for key in properties:
            print("  %s = %s" % (key, str(properties[key])))

        print("frames offset = %u" % frames_offset)
        fp.seek(frames_offset)
        n_frames = 0
        while True:
            pos = fp.tell()

            try:
                frame_start = fp.read(8)
            except Exception as e:
                print("IO error reading %s: %s" % (filename, str(e)))
                break

            if frame_start == b'': # EOF
                break

            (frame_len, frame_hdr_len) = struct.unpack('II', frame_start)

            snappy_frame_header = fp.read(frame_hdr_len)
            frame_header = snappy.decompress(snappy_frame_header)

            frame_props = unpack_properties(frame_header[n_sections * 4:])
            #for key in frame_props:
            #    print("  %s = %s" % (key, str(frame_props[key])))

            section_lengths = []            
            for i in range(0, n_sections):
                off = i * 4;
                (section_length,) = struct.unpack('I', frame_header[off:off+4])
                section_lengths.append(section_length)

            cur_section_off = 0
            for section_length in section_lengths:
                fp.seek(pos + 8 + frame_hdr_len + cur_section_off)
                snappy_section = fp.read(section_length)
                section = snappy.decompress(snappy_section)
                cur_section_off += section_length

            fp.seek(pos + frame_len)
            n_frames += 1

        print("%u frames found" % n_frames)


if __name__ == "__main__":
    main()

