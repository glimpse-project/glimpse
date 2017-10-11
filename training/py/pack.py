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
# This is a python module for reading and writting pack files which are a
# container for frames of training data, optimized for sequentual reads of
# frames with minimal cpu overhead to decompress data on the fly.
#

import sys
import struct
import snappy
import stat
import os
import copy


class PackFrameSection:
    frame = None
    name = None

    _compressed = None
    _decompressed = None

    def __init__(self, frame, name, compressed=None):
        self.frame = frame
        self.name = name
        self._compressed = compressed

    def set_compressed(self, compressed):
        self._decompressed = None
        self._compressed = compressed

    def set_decompressed(self, decompressed):
        self._compressed = None
        self._decompressed = decompressed

    def get_compressed(self):
        return self._compressed;

    def get_decompressed(self):
        if self._decompressed != None:
            return self._decompressed;

        if self._compressed != None:
            self._decompressed = snappy.decompress(self._compressed)
            return self._decompressed

        return None

    def compress(self):
        if self._compressed != None:
            return self._compressed
        if self._decompressed == None:
            return None
        self._compressed = snappy.compress(self._decompressed)
        return self._compressed


class PackFrame:

    pack = None
    properties = {}

    _sections_layout = {}
    _sections = {}

    _file_pos = -1 # -1 implies frame isn't yet part of a pack file on disk
    _frame_len = 0 # > 0 only after being written to disk

    def __init__(self, pack, file_pos=-1):
        self.pack = pack

        if file_pos > 0:
            self._file_pos = file_pos

            fp = pack.fp

            frame_start = fp.read(8)

            if frame_start == b'': # EOF
                raise IOError('no frame to read at end of pack file')

            (self._frame_len, frame_hdr_len) = struct.unpack('II', frame_start)

            snappy_frame_header = fp.read(frame_hdr_len)
            frame_header = snappy.decompress(snappy_frame_header)

            n_sections = len(self.pack.section_names)
            self.properties = Pack._unpack_properties(frame_header[n_sections * 4:])

            cur_section_off = 0
            for i in range(0, n_sections):
                off = i * 4;

                (section_length,) = struct.unpack('I', frame_header[off:off+4])

                section_name = self.pack.section_names[i]
                self._sections_layout[section_name] = {
                        'length': section_length,
                        'offset': file_pos + 8 + frame_hdr_len + cur_section_off
                    }

                cur_section_off += section_length


    def __getitem__(self, key):
        '''Access sections by name'''

        if key in self._sections:
            return self._sections[key]

        n_sections = len(self.pack.section_names)
        for i in range(0, n_sections):
            if self.pack.section_names[i] == key:

                if self._file_pos > 0:
                    section_info = self._sections_layout[key]
                    section_offset = section_info['offset']
                    section_length = section_info['length']
                    fp = self.pack.fp

                    fp.seek(section_offset)
                    compressed = fp.read(section_length)

                    self._sections[key] = PackFrameSection(self, key, compressed)
                    return self._sections[key]
                else:
                    self._sections[key] = PackFrameSection(self, key)
                    return self._sections[key]

        raise ValueError('unknown section name')


    def copy_properties(self):
        '''utility for copying properties from one pack frame to another'''
        return copy.deepcopy(self.properties)


class PackFrameIterator:

    _pack = None
    _current = 0
    _file_pos = 0

    def __init__(self, pack):
        self._pack = pack
        self._file_pos = pack.frames_offset

    def __next__(self):
        if self._pack._is_empty:
            raise StopIteration()

        try:
            self._pack.fp.seek(self._file_pos)
            frame = PackFrame(self._pack, self._file_pos)
            self._file_pos += frame._frame_len
            self._current += 1
            return frame
        except IOError as e:
            raise StopIteration()


class PackProperty:

    type = 'unknown'
    value = 0.0

    def __init__(self, value, property_type = None):
        self.type = property_type
        self.value = value

        if property_type == None:
            if isinstance(value, int) or isinstance(value, long):
                self.type='int64'
            elif isinstance(value, float):
                self.type='double'
            elif isinstance(value, str) or isinstance(value, unicode):
                self.type='string'
            elif isinstance(value, bytearray):
                self.type='blob'
            else:
                raise ValueError("couldn't infer property type from value")


    def get_pack_len(self):
        if self.type == 'string':
            return 24 + len(self.value.encode('utf-8')) + 1
        if self.type == 'blob':
            return 24 + len(self.value)
        else:
            return 32


    def _get_type_int(self):
        if self.type == 'double':
            return 0
        elif self.type == 'int64':
            return 1
        elif self.type == 'string':
            return 2
        elif self.type == 'blob':
            return 3
        else:
            raise Exception('unknown property type')


    def pack(self, buf, offset, name):
        prop_type = self._get_type_int()
        byte_len = self.get_pack_len()
        struct.pack_into('16s', buf, offset, name.encode('utf-8'))
        struct.pack_into('II', buf, offset + 16, prop_type, byte_len)

        if self.type == 'double':
            struct.pack_into('d', buf, offset + 24, self.value)
        elif self.type == 'int64':
            struct.pack_into('q', buf, offset + 24, self.value)
        elif self.type == 'string':
            str_len = byte_len - 24 + 1
            struct.pack_into(str(str_len) + 's', buf, offset + 24, self.value.encode('utf-8'))
        elif self.type == 'blob':
            struct.pack_into(str(len(self.value)) + 's', buf, offset + 24, self.value)
        else:
            raise Exception('unknown property type')

        return byte_len


    def __str__(self):
        return str(self.value)


class Pack:

    filename = None
    fp = None

    major = 1
    minor = 0
    frames_offset = 16 * 1024 * 1024
    properties = {}
    section_names = []

    _is_empty = True # True for newly created pack files


    @staticmethod
    def _unpack_properties(data, max_properties=-1):
        '''helper for decoding pack and frame properties'''

        data_len = len(data)
        off = 0
        properties = {}

        n_properties = 0

        while off < data_len:
            prop_name = data[off:off+16].decode('ascii').split('\0')[0]
            (prop_type, byte_len) = struct.unpack('II', data[off+16:off+24])
            if prop_type == 0: # double
                (val,) = struct.unpack('d', data[off+24:off+32])
                properties[prop_name] = PackProperty(val, 'double')
            elif prop_type == 1: # i64
                (val,) = struct.unpack('q', data[off+24:off+32])
                properties[prop_name] = PackProperty(val, 'int64')
            elif prop_type == 2: # string
                val = data[off+24:off+byte_len-1].decode('utf-8')
                properties[prop_name] = PackProperty(val, 'string')
            elif prop_type == 2: # blob
                val = data[off+24:off+byte_len]
                properties[prop_name] = PackProperty(val, 'blob')
            else:
                pass
                # silently ignore unknown types to maintain forwards
                # compatibility
                # print("ignoring property with unknown type %u" % prop_type)

            off += byte_len

            n_properties += 1
            if n_properties == max_properties:
                break

        return properties


    @staticmethod
    def _measure_properties_pack_size(properties):
        size = 0
        for key in properties:
            prop = properties[key]
            size += prop.get_pack_len()
        return size


    @staticmethod
    def _pack_properties(properties, buf, offset):
        '''helper for encoding pack and frame properties'''

        for key in properties:
            prop = properties[key]
            length = prop.pack(buf, offset, key)
            offset += length


    def __init__(self, filename, writable=False, like=None):
        try:
            mode = os.stat(filename).st_mode
            if not stat.S_ISREG(mode):
                raise ValueError("filename %s doesn't refer to a regular file" % filename)
            exists = True

            if writable:
                self.fp = open(filename, 'r+b')
            else:
                self.fp = open(filename, 'rb')
        except FileNotFoundError as e:
            exists = False
            if not writable:
                raise e

        self.filename = filename

        if exists:
            if like != None:
                raise ValueError("can't re-intilize existing pack like another")

            fp = self.fp

            magic = fp.read(4).decode('ascii')

            if magic != "P4cK":
                raise Exception("%s not a pack file" % filename)

            try:
                (self.frames_offset, hdr_len) = struct.unpack('II', fp.read(8))

                snappy_header = fp.read(hdr_len)
                header = snappy.decompress(snappy_header)

                (self.major, self.minor, part0_size, n_properties,
                 n_sections) = struct.unpack('5I', header[:20])

                if self.major != 1:
                    raise ValueError("Unsupported pack file version")

                for n in range(0, n_sections):
                    off = 20 + n * 64
                    section_name = header[off:off+64].decode('ascii').split('\0')[0]
                    self.section_names.append(section_name)

                self.properties = Pack._unpack_properties(header[20 + n_sections * 64:],
                                                          max_properties=n_properties)
                self._is_empty = False
            except struct.error as e:
                raise Exception('corrupt, truncated pack file')
        else:
            self._is_empty = True


    def write_header(self):

        if not self.fp:
            self.fp = open(self.filename, 'w+b')

        fp = self.fp

        # header layout:
        # 5I: major, minor, part0_size, n_properties, n_sections
        # 64s * n_sections
        # packed properties

        n_sections = len(self.section_names)
        part0_size = 20 + 64 * n_sections

        props_size = self._measure_properties_pack_size(self.properties)
        header_len = part0_size + props_size
        header_buf = bytearray(header_len)

        n_properties = len(self.properties)

        struct.pack_into('5I', header_buf, 0,
                         self.major, self.minor,
                         part0_size,
                         n_properties,
                         n_sections)

        for n in range(0, n_sections):
            off = 20 + n * 64
            struct.pack_into('64s', header_buf, off, self.section_names[n].encode('utf-8'))

        self._pack_properties(self.properties,
                              header_buf,
                              part0_size)

        fp.seek(0)

        # XXX: it's really pretty suprising that Python has no zero-copy way of
        # casting a bytearray to bytes or a readonly memoryview and you can't
        # fake it by subclassing the builtin memoryview class. We have to pass
        # in a read-only bytes like object here which means we have to incur a
        # redundant copy :(
        compressed = snappy.compress(bytes(header_buf))
        compressed_len = len(compressed)

        if compressed_len > self.frames_offset:
            # If this is a new file then there's no inherent limit to the
            # size of the header, but otherwise we can't grew beyond the
            # the space allocated when the file was first written:
            if self._is_empty:
                while compressed_len < self.frames_offset:
                    self.frames_offset *= 2
            else:
                raise Exception("header too long")

        fp.write(struct.pack('4sII', 'P4cK'.encode('ascii'), self.frames_offset, compressed_len))

        fp.write(compressed)
        
        if fp.tell() < self.frames_offset:
            fp.seek(self.frames_offset - 1)
            fp.write(b' ')


    def copy_properties(self):
        '''utility for copying properties from one pack to another'''
        return copy.deepcopy(self.properties)


    def new_frame(self):
        '''Convenience for constructing a frame associated with this pack'''
        return PackFrame(self)


    def append_frame(self, frame):
        '''writes the given frame to the end of the pack file'''

        n_sections = len(self.section_names)
        props_size = self._measure_properties_pack_size(frame.properties)

        buf = bytearray(4 * n_sections + props_size)

        total_length = 8

        sections = []
        for n in range(0, n_sections):
            section_name = self.section_names[n]
            section = frame[section_name]

            compressed = section.compress()

            sections.append(compressed)
            if compressed != None:
                section_length = len(compressed)
            else:
                section_length = 0
            struct.pack_into('I', buf, n * 4, section_length)

            total_length += section_length
        
        self._pack_properties(frame.properties, buf, n_sections * 4)

        header = snappy.compress(bytes(buf))
        header_length = len(header)

        total_length += header_length

        if not self.fp:
            self.write_header()

        fp = self.fp
        fp.seek(0, 2)
        fp.write(struct.pack('II', total_length, header_length))
        fp.write(header)

        for n in range(0, n_sections):
            compressed = sections[n]
            if compressed != None:
                fp.write(compressed)


    def __iter__(self):
        return PackFrameIterator(self)


class PackReader(Pack):

    def __init__(self, filename):
        Pack.__init__(self, filename)


class PackWriter(Pack):

    def __init__(self, filename):
        Pack.__init__(self, filename, writable=True)



