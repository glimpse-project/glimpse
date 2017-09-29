/*
 * Copyright (C) 2017 Kwamecorp
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

#define _BSD_SOURCE 1 // for strdup

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <snappy-c.h>

#include "xalloc.h"
#include "llist.h"
#include "pack.h"

#define MAGIC_STRING "pack"

static struct property *
lookup_property(LList *properties, const char *name)
{
    for (LList *l = properties; l; l = l->next) {
        struct property *prop = (struct property *)l->data;
        if (strcmp(prop->name, name) == 0)
            return prop;
    }
    return NULL;
}

static struct property *
get_property(LList *properties, const char *name, char **err)
{
    struct property *prop = lookup_property(properties, name);
    if (err)
        *err = NULL;
    if (!prop)
        xasprintf(err, "property %s not found", name);
    return prop;
}

static void
remove_property(LList *properties, const char *name)
{
    for (LList *l = properties; l; l = l->next) {
        struct property *prop = (struct property *)l->data;
        if (strcmp(prop->name, name) == 0) {
            llist_free(llist_remove(l), NULL, NULL);
            free(prop);
        }
    }
}

static LList *
_set_i64(LList *properties, const char *name, int64_t val)
{
    struct property *prop;

    remove_property(properties, name);

    uint32_t byte_len = sizeof(*prop);
    prop = (struct property *)xcalloc(1, byte_len);

    assert(strlen(name) < sizeof(prop->name));

    strncpy(prop->name, name, sizeof(prop->name));
    prop->type = PROP_INT64;
    prop->i64_val = val;
    prop->byte_len = byte_len;

    return llist_insert_before(properties, llist_new(prop));
}

void
pack_set_i64(struct pack_file *file, const char *name, int64_t val)
{
    file->properties = _set_i64(file->properties, name, val);
}

void
pack_frame_set_i64(struct pack_frame *frame, const char *name, int64_t val)
{
    frame->properties = _set_i64(frame->properties, name, val);
}

static LList *
_set_double(LList *properties, const char *name, double val)
{
    struct property *prop;

    remove_property(properties, name);

    uint32_t byte_len = sizeof(*prop);
    prop = (struct property *)xcalloc(1, byte_len);

    assert(strlen(name) < sizeof(prop->name));

    strncpy(prop->name, name, sizeof(prop->name));
    prop->type = PROP_DOUBLE;
    prop->double_val = val;
    prop->byte_len = byte_len;

    return llist_insert_before(properties, llist_new(prop));
}

void
pack_set_double(struct pack_file *file, const char *name, double val)
{
    file->properties = _set_double(file->properties, name, val);
}

void
pack_frame_set_double(struct pack_frame *frame, const char *name, double val)
{
    frame->properties = _set_double(frame->properties, name, val);
}

static LList *
_set_blob(LList *properties,
          const char *name,
          const uint8_t *data,
          uint32_t len)
{
    struct property *prop;

    assert(strlen(name) < sizeof(prop->name));

    remove_property(properties, name);

    uint32_t byte_len = sizeof(*prop) + len;
    prop = (struct property *)xcalloc(1, byte_len);

    strncpy(prop->name, name, sizeof(prop->name));
    prop->type = PROP_BLOB;
    memcpy(prop->blob, data, len);
    prop->byte_len = byte_len;

    return llist_insert_before(properties, llist_new(prop));
}

void
pack_set_blob(struct pack_file *file,
              const char *name,
              const uint8_t *data,
              uint32_t len)
{
    file->properties = _set_blob(file->properties, name, data, len);
}

void
pack_frame_set_blob(struct pack_frame *frame,
                    const char *name,
                    const uint8_t *data,
                    uint32_t len)
{
    frame->properties = _set_blob(frame->properties, name, data, len);
}

static LList *
_set_string(LList *properties,
            const char *name,
            const char *string)
{
    struct property *prop;

    assert(strlen(name) < sizeof(prop->name));

    remove_property(properties, name);

    uint32_t len = strlen(string) + 1;
    uint32_t byte_len = sizeof(*prop) + len;
    prop = (struct property *)xcalloc(1, byte_len);

    strncpy(prop->name, name, sizeof(prop->name));
    prop->type = PROP_STRING;
    memcpy(prop->blob, string, len);
    prop->byte_len = byte_len;

    return llist_insert_before(properties, llist_new(prop));
}

void
pack_set_string(struct pack_file *file, const char *name, const char *val)
{
    file->properties = _set_string(file->properties, name, val);
}

void
pack_frame_set_string(struct pack_frame *frame, const char *name, const char *val)
{
    frame->properties = _set_string(frame->properties, name, val);
}

static int64_t
_get_i64(LList *properties, const char *name, char **err)
{
    struct property *prop = get_property(properties, name, err);

    if (prop) {
        if (prop->type != PROP_INT64) {
            xasprintf(err, "property type mismatch");
            return 0;
        }

        return prop->i64_val;
    } else
        return 0;
}

int64_t
pack_get_i64(struct pack_file *pack, const char *name, char **err)
{
    return _get_i64(pack->properties, name, err);
}

int64_t
pack_frame_get_i64(struct pack_frame *frame, const char *name, char **err)
{
    return _get_i64(frame->properties, name, err);
}

static double
_get_double(LList *properties, const char *name, char **err)
{
    struct property *prop = get_property(properties, name, err);

    if (prop) {
        if (prop->type != PROP_DOUBLE) {
            xasprintf(err, "property type mismatch");
            return 0;
        }

        return prop->double_val;
    } else
        return 0;
}

double
pack_get_double(struct pack_file *pack, const char *name, char **err)
{
    return _get_double(pack->properties, name, err);
}

double
pack_frame_get_double(struct pack_frame *frame, const char *name, char **err)
{
    return _get_double(frame->properties, name, err);
}

static const char *
_get_string(LList *properties, const char *name, char **err)
{
    struct property *prop = get_property(properties, name, err);

    if (prop) {
        if (prop->type != PROP_BLOB) {
            xasprintf(err, "property type mismatch");
            return NULL;
        }

        return (const char *)prop->blob;
    } else
        return NULL;
}

const char *
pack_get_string(struct pack_file *pack, const char *name, char **err)
{
    return _get_string(pack->properties, name, err);
}

const char *
pack_frame_get_string(struct pack_frame *frame, const char *name, char **err)
{
    return _get_string(frame->properties, name, err);
}

/* The order of calls determines the in-file order of sections */
void
pack_declare_frame_section(struct pack_file *file, const char *name)
{
    file->section_names = (char **)xrealloc(file->section_names,
                                            sizeof(char *) * (file->n_sections + 1));
    file->section_names[file->n_sections++] = strdup(name);
}

void
pack_close(struct pack_file *file)
{
    if (file->fp)
        fclose(file->fp);

    for (LList *l = file->properties; l; l = l->next) {
        free(l->data);
    }
    llist_free(file->properties, NULL, NULL);

    for (int i = 0; i < file->n_sections; i++)
        free(file->section_names[i]);
    free(file->section_names);

    free(file);
}

/* Opens the file and decodes the header
 */
struct pack_file *
pack_open(const char *filename, char **err)
{
    struct pack_file *pack = NULL;
    struct stat st;

    bool preexists = (stat(filename, &st) == 0);

    FILE *fp = NULL;

    if (preexists) {
        char magic[sizeof(MAGIC_STRING)];
        uint32_t compressed_header_size = 0;
        struct file_header_part0 *part0;

        fp = fopen(filename, "r+");
        if (!fp) {
            xasprintf(err, "Failed to open %s\n", filename);
            goto error;
        }

        fseek(fp, 0, SEEK_SET);

        if (fread(magic, sizeof(magic), 1, fp) != 1) {
            xasprintf(err, "Failed to read pack magic marker");
            goto error;
        }

        if (fread(&compressed_header_size, 4, 1, fp) != 1) {
            xasprintf(err, "Failed to read size of header");
            goto error;
        }

        uint8_t *compressed_header = (uint8_t *)xmalloc(compressed_header_size);
        if (fread(compressed_header, compressed_header_size, 1, fp) != 1) {
            xasprintf(err, "Failed to read compressed header");
            goto error;
        }
        size_t header_size;
        if (snappy_uncompressed_length((char *)compressed_header,
                                       compressed_header_size,
                                       &header_size)
            != SNAPPY_OK)
        {
            xasprintf(err, "Failed to query uncompressed size of header");
            free(compressed_header);
            goto error;
        }
        uint8_t *header = (uint8_t *)xmalloc(header_size);
        if (snappy_uncompress((char *)compressed_header,
                              compressed_header_size,
                              (char *)header, &header_size) != SNAPPY_OK) {
            xasprintf(err, "Failed to uncompress header");
            free(compressed_header);
            free(header);
            goto error;
        }

        part0 = (struct file_header_part0 *)header;

        pack = (struct pack_file *)xcalloc(1, sizeof(*pack));
        pack->fp = fp;
        pack->guard_band = part0->frames_offset;

        for (unsigned i = 0; i < part0->n_sections; i++)
            pack_declare_frame_section(pack, part0->section_names[i]);

        struct property *prop = (struct property *)(header + part0->part0_size);
        for (unsigned i = 0; i < part0->n_properties; i++) {
            switch (prop->type) {
            case PROP_INT64:
                pack_set_i64(pack, prop->name, prop->i64_val);
                break;
            case PROP_DOUBLE:
                pack_set_double(pack, prop->name, prop->double_val);
                break;
            case PROP_STRING:
                pack_set_string(pack, prop->name, (char *)prop->blob);
                break;
            case PROP_BLOB:
                {
                    unsigned blob_len =
                        prop->byte_len - offsetof(struct property, blob);

                    pack_set_blob(pack, prop->name, prop->blob, blob_len);
                    break;
                }
            };

            prop = (struct property *)((uint8_t *)prop + prop->byte_len);
        }
        if (!pack_get_i64(pack, "width", err) ||
            !pack_get_i64(pack, "height", err))
        {
            xasprintf(err, "Failed to find width or height property");
            free(compressed_header);
            free(header);
            goto error;
        }
    } else {
        fp = fopen(filename, "w+");
        if (!fp) {
            xasprintf(err, "Failed to open %s\n", filename);
            goto error;
        }

        pack = (struct pack_file *)xcalloc(1, sizeof(*pack));
        pack->fp = fp;

        /* The default amount of space we have for header data before
         * frames start
         */
        pack->guard_band = 16 * 1024 * 1024;
        ftruncate(fileno(pack->fp), pack->guard_band);
    }

    fseek(pack->fp, pack->guard_band, SEEK_SET);
    pack->frame_cursor = 0;

    return pack;

error:
    if (pack) {
        for (LList *l = pack->properties; l; l = l->next) {
            free(l->data);
        }
        free(pack);
    }
    if (fp)
        fclose(fp);
    return NULL;
}

static uint32_t
get_properties_pack_size(LList *properties)
{
    uint32_t props_pos = 0;

    for (LList *l = properties; l; l = l->next) {
        struct property *prop = (struct property *)l->data;

        props_pos += prop->byte_len;
    }

    return props_pos;
}

static uint32_t
append_properties(LList *properties,
                  uint8_t *dest,
                  unsigned len)
{
    uint32_t props_pos = 0;

    if (get_properties_pack_size(properties) > len)
        return 0;

    for (LList *l = properties; l; l = l->next) {
        struct property *prop = (struct property *)l->data;

        memcpy(dest + props_pos, prop, prop->byte_len);

        props_pos += prop->byte_len;
    }

    return props_pos;
}

bool
pack_write_header(struct pack_file *pack, char **err)
{
    struct file_header_part0 *part0;
    uint32_t part0_size = 
        sizeof(struct file_header_part0) +
        pack->n_sections * sizeof(part0->section_names[0]);
    uint32_t props_len = get_properties_pack_size(pack->properties);
    uint32_t uncompressed_header_size = part0_size + props_len;
    uint8_t *uncompressed_header = (uint8_t *)xcalloc(1, uncompressed_header_size);

    uint32_t size;

    uint8_t *guard_band = NULL;

    part0 = (struct file_header_part0 *)uncompressed_header;
    part0->major_version = 1;
    part0->minor_version = 0;
    part0->part0_size = part0_size;
    part0->frames_offset = pack->guard_band;
    part0->n_properties = llist_length(pack->properties);

    part0->n_sections = pack->n_sections;
    for (int i = 0; i < pack->n_sections; i++)
        strncpy(part0->section_names[i], pack->section_names[i], 64);

    if (part0->n_properties) {
        append_properties(pack->properties,
                          uncompressed_header + part0_size,
                          props_len);
    }

    size_t compressed_size =
        snappy_max_compressed_length(uncompressed_header_size);
    uint8_t *compressed_header = (uint8_t *)xcalloc(1, compressed_size);

    if (snappy_compress((char *)uncompressed_header,
                        uncompressed_header_size,
                        (char *)compressed_header,
                        &compressed_size) != SNAPPY_OK)
    {
        xasprintf(err, "snappy_compress() error for pack file header");
        goto error;
    }

    if (sizeof(MAGIC_STRING) + 4 + compressed_size > pack->guard_band) {
        xasprintf(err, "Header doesn't fit in pack file guard band (%u bytes)",
                  (unsigned)pack->guard_band);
        goto error;
    }

    guard_band = (uint8_t *)xcalloc(1, pack->guard_band);

    memcpy(guard_band, MAGIC_STRING, sizeof(MAGIC_STRING));
    size = compressed_size;
    memcpy(guard_band + sizeof(MAGIC_STRING), &size, 4);
    memcpy(guard_band + sizeof(MAGIC_STRING) + 4, compressed_header, size);

    fseek(pack->fp, 0, 0);

    if (fwrite(guard_band, pack->guard_band, 1, pack->fp) != 1) {
        xasprintf(err, "Failed to write pack file header\n");
        goto error;
    }

    fseek(pack->fp, 0, 2);
    return true;

error:
    fseek(pack->fp, 0, 2);
    free(uncompressed_header);
    free(compressed_header);
    free(guard_band);
    return false;
}

struct pack_frame *
pack_frame_new(struct pack_file *pack)
{
    struct pack_frame *frame = (struct pack_frame *)
        xcalloc(1, sizeof(struct pack_frame) +
                sizeof(frame->sections[0]) * pack->n_sections);

    frame->pack = pack;

    return frame;
}

/* Note the data isn't copied so the caller needs to make sure it's kept
 * at least until pack_frame_compress() is called
 */
void
pack_frame_set_section(struct pack_frame *frame,
                       const char *section,
                       uint8_t *data,
                       size_t len)
{
    int i;
    for (i = 0; i < frame->pack->n_sections; i++) {
        if (strcmp(section, frame->pack->section_names[i]) == 0)
            break;
    }
    if (i == frame->pack->n_sections) {
        fprintf(stderr, "Section name \"%s\" isn't defined in pack\n", section);
        exit(1);
    }

    frame->sections[i].uncompressed_size = len;
    frame->sections[i].uncompressed_data = data;
}

uint32_t
pack_frame_compress(struct pack_frame *frame, char **err)
{
    uint32_t compressed_bytes = 0;
    size_t compressed_header_size;
    char *snappy_header;

    if (frame->total_length) {
        xasprintf(err, "frame already compressed\n");
        return 0;
    }

    /* The header is an array of u32 lengths for each section followed by a
     * series of properties
     */
    uint32_t uncompressed_header_size = sizeof(uint32_t) * frame->pack->n_sections;

    uint32_t props_len = get_properties_pack_size(frame->properties);
    uncompressed_header_size += props_len;

    uint8_t *header = (uint8_t *)alloca(uncompressed_header_size);
    uint32_t *section_lengths = (uint32_t *)header;

    /* Each frame starts with a u32 frame length + u32 compressed header len */
    frame->total_length = 8;
    compressed_bytes = uncompressed_header_size;

    for (int i = 0; i < frame->pack->n_sections; i++) {
        uint32_t uncompressed_size = frame->sections[i].uncompressed_size;
        size_t compressed_size = snappy_max_compressed_length(uncompressed_size);
        char *snappy_dest = (char *)xmalloc(compressed_size);

        if (snappy_compress((char *)frame->sections[i].uncompressed_data,
                            uncompressed_size,
                            snappy_dest, &compressed_size) != SNAPPY_OK)
        {
            char *ignore0 = NULL, *ignore1 = NULL;
            xasprintf(err, "snappy_compress() error for %s/%u section %s\n",
                      pack_frame_get_string(frame, "mocap-name", &ignore0),
                      (unsigned)pack_frame_get_i64(frame, "mocap-frame", &ignore1),
                      frame->pack->section_names[i]);
            free(ignore0);
            free(ignore1);
            goto error;
        }
        frame->sections[i].compressed_size = compressed_size;
        frame->sections[i].compressed_data = (uint8_t *)snappy_dest;
        //frame->sections[i].uncompressed_size = 0;
        //frame->sections[i].uncompressed_data = NULL;

        section_lengths[i] = compressed_size;
        frame->total_length += compressed_size;
        compressed_bytes += uncompressed_size;
    }

    append_properties(frame->properties,
                      header + sizeof(uint32_t) * frame->pack->n_sections,
                      props_len);

    compressed_header_size =
        snappy_max_compressed_length(uncompressed_header_size);
    snappy_header = (char *)xmalloc(compressed_header_size);

    if (snappy_compress((char *)header, uncompressed_header_size,
                        snappy_header, &compressed_header_size) != SNAPPY_OK)
    {
        char *ignore0 = NULL, *ignore1 = NULL;
        xasprintf(err, "snappy_compress() error for %s/%u header\n",
                  pack_frame_get_string(frame, "mocap-name", &ignore0),
                  (unsigned)pack_frame_get_i64(frame, "mocap-frame", &ignore1));
        free(ignore0);
        free(ignore1);
        goto error;
    }

    frame->compressed_header_size = compressed_header_size;
    frame->compressed_header = (uint8_t *)snappy_header;


    frame->total_length += compressed_header_size;

    return compressed_bytes;

error:
    for (int i = 0; i < frame->pack->n_sections; i++) {
        free(frame->sections[i].compressed_data);
        frame->sections[i].compressed_data = NULL;
    }

    frame->compressed_header_size = 0;
    free(frame->compressed_header);
    frame->compressed_header = NULL;

    frame->total_length = 0;

    return 0;
}

void
pack_frame_free(struct pack_frame *frame)
{
    for (LList *l = frame->properties; l; l = l->next)
        free(l->data);
    llist_free(frame->properties, NULL, NULL);

    for (int i = 0; i < frame->pack->n_sections; i++)
        free(frame->sections[i].compressed_data);

    free(frame->compressed_header);

    free(frame);
}

static void
xwrite(FILE *fp, uint8_t *buf, int len)
{
    if (fwrite(buf, 1, len, fp) != (unsigned)len) {
        fprintf(stderr, "IO error while writing: %m\n");
        exit(1);
    }
}

uint32_t
pack_append_frame(struct pack_file *file, struct pack_frame *frame)
{
    fseek(file->fp, 0, SEEK_END);

    xwrite(file->fp, (uint8_t *)&frame->total_length, 4);

    xwrite(file->fp, (uint8_t *)&frame->compressed_header_size, 4);
    xwrite(file->fp, (uint8_t *)frame->compressed_header,
           frame->compressed_header_size);

    for (int i = 0; i < file->n_sections; i++) {
        xwrite(file->fp, (uint8_t *)frame->sections[i].compressed_data,
               frame->sections[i].compressed_size);
    }

    return frame->total_length;
}

struct pack_frame *
pack_read_frame(struct pack_file *pack, int n, char **err)
{
    struct pack_frame *frame = NULL;

    if (pack->frame_cursor != n) {
        fseek(pack->fp, pack->guard_band, SEEK_SET);
        for (int i = 0; i < n; i++) {
            uint32_t frame_len;
            if (fread(&frame_len, 4, 1, pack->fp) != 1) {
                xasprintf(err, "Failed to read frame %d length", i);
                return NULL;
            }
            fseek(pack->fp, frame_len, SEEK_CUR);
        }
        pack->frame_cursor = n;
    }

    long pos = ftell(pack->fp);
    uint32_t frame_len = 0;
    uint32_t compressed_header_size;
    uint8_t *compressed_header = NULL;
    size_t header_size;
    uint8_t *header = NULL;
    struct property *prop;

    if (fread(&frame_len, 4, 1, pack->fp) != 1) {
        xasprintf(err, "Failed to read frame length");
        return NULL;
    }

    frame = (struct pack_frame *)xcalloc(1, sizeof(*frame) +
                                         pack->n_sections * sizeof(frame->sections[0]));
    frame->pack = pack;

    if (fread(&compressed_header_size, 4, 1, pack->fp) != 1) {
        xasprintf(err, "Failed to read size of header");
        goto error;
    }

    compressed_header = (uint8_t *)xmalloc(compressed_header_size);
    if (fread(compressed_header, compressed_header_size, 1, pack->fp) != 1) {
        xasprintf(err, "Failed to read compressed header");
        goto error;
    }
    if (snappy_uncompressed_length((char *)compressed_header,
                                   compressed_header_size,
                                   &header_size)
        != SNAPPY_OK)
    {
        xasprintf(err, "Failed to query uncompressed size of header");
        goto error;
    }
    header = (uint8_t *)xmalloc(header_size);
    if (snappy_uncompress((char *)compressed_header,
                          compressed_header_size,
                          (char *)header, &header_size) != SNAPPY_OK) {
        xasprintf(err, "Failed to uncompress header");
        goto error;
    }

    prop = (struct property *)(header + 4 * pack->n_sections);
    while ((uint8_t *)prop < (header + header_size)) {
        switch (prop->type) {
        case PROP_INT64:
            pack_set_i64(pack, prop->name, prop->i64_val);
            break;
        case PROP_DOUBLE:
            pack_set_double(pack, prop->name, prop->double_val);
            break;
        case PROP_STRING:
            pack_set_string(pack, prop->name, (char *)prop->blob);
            break;
        case PROP_BLOB:
            {
                unsigned blob_len =
                    prop->byte_len - offsetof(struct property, blob);

                pack_set_blob(pack, prop->name, prop->blob, blob_len);
                break;
            }
        };

        prop = (struct property *)((uint8_t *)prop + prop->byte_len);
    }

    for (int i = 0; i < pack->n_sections; i++) {
        frame->sections[i].compressed_size = *(uint32_t *)(header + i * 4);
        frame->sections[i].compressed_data = (uint8_t *)
            xmalloc(frame->sections[i].compressed_size);

        if (fread(frame->sections[i].compressed_data,
                  frame->sections[i].compressed_size, 1, pack->fp) != 1)
        {
            xasprintf(err, "Failed to read compressed section %s",
                      pack->section_names[i]);
            goto error;
        }
    }

    fseek(pack->fp, pos + frame_len, SEEK_SET);
    pack->frame_cursor = n + 1;
    return frame;

error:
    for (int i = 0; i < pack->n_sections; i++) {
        free(frame->sections[i].compressed_data);
    }
    free(compressed_header);
    free(header);
    free(frame);
    return NULL;
}

uint8_t *
pack_frame_get_section(struct pack_frame *frame,
                       const char *section,
                       uint32_t *len,
                       char **err)
{
    struct pack_file *pack = frame->pack;

    *len = 0;

    for (int i = 0; i < pack->n_sections; i++) {
        size_t section_len;

        if (strcmp(frame->pack->section_names[i], section) != 0)
            continue;

        if (frame->sections[i].compressed_data) {
            xasprintf(err, "Frame needs to be read via pack_read_frame() first");
            return NULL;
        }

        if (snappy_uncompressed_length((char *)frame->sections[i].compressed_data,
                                       frame->sections[i].compressed_size,
                                       &section_len)
            != SNAPPY_OK)
        {
            xasprintf(err, "Failed to query uncompressed size of section %s",
                      pack->section_names[i]);
            return NULL;
        }
        uint8_t *section = (uint8_t *)xmalloc(section_len);
        if (snappy_uncompress((char *)frame->sections[i].compressed_data,
                              frame->sections[i].compressed_size,
                              (char *)section, &section_len) != SNAPPY_OK)
        {
            xasprintf(err, "Failed to uncompress section");
            free(section);
            return NULL;
        }

        frame->sections[i].uncompressed_size = section_len;
        frame->sections[i].uncompressed_data = section;

        *len = section_len;
        return section;
    }

    return NULL;
}
