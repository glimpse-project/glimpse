/*
 * Copyright (C) 2017 Glimp IP Ltd
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once


/*
 * Pack files are intended to be an efficient way of managing large
 * streams of training data organised logically into 'frames'.
 *
 * Frames are split into 'sections' so e.g. their might be a "depth"
 * buffer section, a color "labels" section and a miscellaneous "json"
 * meta data section for each frame.
 *
 * Each section is separately compressed using Snappy to help reduce
 * IO while reading data but minimize CPU overhead before being able
 * to process the data.
 *
 * The format supports some simple (key, value) properties that may
 * be either associated with the pack file or individual frames. (int64,
 * double, string or an arbitrary 'blob' of data)
 *
 * Some pack file requirements:
 *
 * - Want to be able to append frames to pack file without re-writing exiting
 *   frames
 * - Want to be able to add/edit some properties without re-writing entire
 *   file
 * - It should be easy to create a new pack file based on an existing file
 *   but adding/removing/editing the sections of each frame.
 * - If writing a new file with additional sections then it should be possible
 *   to easily copy across unknown sections from the previous file without
 *   uncompressing or decoding them.
 * - It should be straight forward to be able to shuffle a pack file.
 *
 *
 * API Overview
 *
 * There are two top-level structures exposed by the api: a 'pack' and a 'frame'
 *
 * A 'pack' represents the whole file, a 'frame' represents an individual frame
 * with associated sections.
 *
 * Error handling is managed via char **err pointers that can return an error
 * string if a function fails. Passing NULL can be compared to not catching an
 * exception and a failure will print a message and abort the program.
 *
 * struct pack_file *pack = pack_open('foo.pack', NULL);
 * pack_set_i64(pack, "width", 172);
 * pack_set_i64(pack, "height", 224);
 * pack_declare_frame_section(pack, "depth");
 * pack_declare_frame_section(pack, "labels");
 * pack_write_header(pack, NULL);
 *
 * struct pack_frame *frame = pack_frame_new(pack);
 * pack_frame_set_section("depth", depth_data, depth_len);
 * pack_frame_set_section("labels", label_data, label_len);
 * pack_frame_set_string(frame, "mocap-name", "flibble");
 * pack_frame_set_i64(frame, "mtime", sb.st_mtim.tv_sec);
 * pack_frame_compress(frame);
 * pack_append_frame(pack, frame);
 * pack_frame_free(frame);
 *
 * pack_close(pack);
 */


/*
 * At a glance a pack file is laid out like this:
 *
 * char magic[4]: "P4cK" (not nul terminated)
 * u32 frames_offset
 * u32 compressed_header_len
 * snappy compressed {
 *      file_header_part0 {
 *          major
 *          minor
 *          part0_size
 *          n_properties
 *          n_sections
 *          section_names[][64] {
 *          }
 *      }
 *      properties[n_properties] {
 *      }
 * }
 *
 * (By default the above (after compression) has to fit within 16MB, whereby
 * the header can be edited so long as it doesn't exceed 16MB.)
 *
 * frames[] {
 *      frame_len
 *      header_len
 *      header {
 *      }
 *      sections[] {
 *      }
 * }
 *
 */


enum property_type {
    PROP_DOUBLE,
    PROP_INT64,
    PROP_STRING,
    PROP_BLOB
};

#define BASE_PROPERTY \
    char name[16]; \
    uint32_t type; \
    uint32_t byte_len

struct property {
    BASE_PROPERTY;
};
struct double_property {
    BASE_PROPERTY;
    double double_val;
};
struct int64_property {
    BASE_PROPERTY;
    int64_t i64_val;
};
struct string_property {
    BASE_PROPERTY;
    uint8_t string[];
};
struct blob_property {
    BASE_PROPERTY;
    uint8_t blob[];
};

/* This will always be the first structure found at the beginning of the file
 *
 * In the future if we want to extend the header information then minor_version
 * bumps could imply that this header and properties are followed by a
 * struct header_part2 {} with more data.
 *
 */
struct file_header_part0
{
    /* Bumped for incompatible struct layout changes */
    uint32_t major_version;
    /* Bumped for extended header structs (backwards compatible) */
    uint32_t minor_version;

    /* For skipping ahead to the properties that follow this structure */
    uint32_t part0_size;

    /* After this header part0 and the section names there is an array of
     * properties
     */
    uint32_t n_properties;

    /* These part0 fields are followed by an array of this many (64 byte)
     * section names.  The number of sections specified here also determines
     * the how many section lengths will be at the start of a frame header.
     */
    uint32_t n_sections;

    char section_names[][64];
};

/* On disk a frame looks like this:
 *
 * u32: total frame size on disk (this included)
 * u32: compressed frame_header size
 * snappy compressed {
 *   u32 section_lengths[n_sections]
 *   properties[] {
 *   }
 * }
 * snappy compressed {
 *   section 0
 * }
 * snappy compressed {
 *   section 1
 * }
 * snappy compressed {
 *   section 2
 * }
 *
 * Each section is compressed separately
 */


/* For more conveniently describing a file in-memory before serializing */
struct pack_file {
    FILE *fp;

    LList *properties;

    int n_sections;
    char **section_names;

    size_t guard_band;

    int frame_cursor;
};

struct pack_frame {
    struct pack_file *pack;

    uint32_t total_length;// set when compressed

    uint32_t compressed_header_size;
    uint8_t *compressed_header;

    LList *properties;

    /* Length will match pack->n_sections */
    struct {
        uint32_t uncompressed_size;
        uint8_t *uncompressed_data;
        uint32_t compressed_size;
        uint8_t *compressed_data;
    } sections[];
};



#ifdef __cplusplus
extern "C" {
#endif

/* Opens a pack file and decodes the header
 */
struct pack_file *pack_open(const char *filename, char **err);

/* Closes a pack file and frees the pack_file structure */
void pack_close(struct pack_file *file);

/* Declares a per-frame section. Every frame in the pack file has the same
 * set of associated sections.
 */
void pack_declare_frame_section(struct pack_file *file, const char *name);

void pack_set_i64(struct pack_file *file, const char *name, int64_t val);
void pack_set_double(struct pack_file *file, const char *name, double val);
void pack_set_blob(struct pack_file *file,
                   const char *name,
                   const uint8_t *data,
                   uint32_t len);
void pack_set_string(struct pack_file *file, const char *name, const char *val);
int64_t pack_get_i64(struct pack_file *pack, const char *name, char **err);
double pack_get_double(struct pack_file *pack, const char *name, char **err);
const char *pack_get_string(struct pack_file *pack, const char *name, char **err);
const uint8_t *pack_get_blob(struct pack_file *pack,
                             const char *name,
                             uint32_t *len,
                             char **err);

bool pack_write_header(struct pack_file *pack, char **err);

struct pack_frame *pack_read_frame(struct pack_file *pack, int n, char **err);

uint32_t pack_append_frame(struct pack_file *file, struct pack_frame *frame);


/*
 * Returns a frame proxy that can be configured and then appended to the file.
 */
struct pack_frame *pack_frame_new(struct pack_file *pack);

/* Note the uncompressed section data isn't copied by this function so the
 * caller needs to make sure it's kept at least until pack_frame_compress() is
 * called
 */
void pack_frame_set_section(struct pack_frame *frame,
                            const char *section,
                            uint8_t *data,
                            size_t len);

void pack_frame_set_i64(struct pack_frame *frame, const char *name, int64_t val);
void pack_frame_set_double(struct pack_frame *frame, const char *name, double val);
void pack_frame_set_blob(struct pack_frame *frame,
                         const char *name,
                         const uint8_t *data,
                         uint32_t len);
void pack_frame_set_string(struct pack_frame *frame, const char *name, const char *val);
int64_t pack_frame_get_i64(struct pack_frame *frame, const char *name, char **err);
double pack_frame_get_double(struct pack_frame *frame, const char *name, char **err);
const char *pack_frame_get_string(struct pack_frame *frame, const char *name, char **err);

uint32_t pack_frame_compress(struct pack_frame *frame, char **err);

uint8_t *pack_frame_get_section(struct pack_frame *frame,
                                const char *section,
                                uint32_t *len,
                                char **err);

void pack_frame_free(struct pack_frame *frame);

#ifdef __cplusplus
}; // "C"
#endif
