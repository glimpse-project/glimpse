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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <dirent.h>
#include <stdint.h>
#include <libgen.h>
#include <getopt.h>

#include "tinyexr.h"

#include "half.hpp"

#ifdef DEBUG
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

using half_float::half;

static bool write_half_float = false;

static void *
xmalloc(size_t size)
{
    void *ret = malloc(size);
    if (ret == NULL)
        exit(1);
    return ret;
}

enum image_format {
    IMAGE_FORMAT_X8,
    IMAGE_FORMAT_XFLOAT,
    IMAGE_FORMAT_XHALF,
};

struct image
{
    enum image_format format;
    int width;
    int height;
    int stride;

    union {
        uint8_t *data_u8;
        float *data_float;
        half *data_half;
    };
};

static struct image *
load_pfm(const char *filename)
{
    const char *asset_buf = NULL;
    long asset_buf_len;

    struct stat sb;

    int line_end[3] = { 0, 0, 0 };
    const char *type_line;
    const char *resolution_line;
    const char *scale_line;

    char *width_end = NULL;
    char *height_end = NULL;
    char *scale_end = NULL;

    int width, height;
    double scale;

    char *pfm_data_start;
    long pfm_data_len;

    struct image *ret = NULL;


    //return_if_fail(asset_manager != NULL, NULL);
    //return_if_fail(filename != NULL, NULL);

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %m\n", filename);
        return NULL;
    }

    fstat(fd, &sb);

    asset_buf = (const char *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (asset_buf) {
        asset_buf_len = sb.st_size;
        debug("Mapped PFM file %p, len = %ld\n", asset_buf, asset_buf_len);
    } else {
        fprintf(stderr, "Failed to map PFM file\n");
        goto error_mmap;
    }

    int i, j;
    for (i = 0, j = 0; i < asset_buf_len && j < 3; i++) {
        if (asset_buf[i] == '\n')
            line_end[j++] = i;
    }
    if (i == asset_buf_len) {
        fprintf(stderr, "Parsing PFM header reached end of file\n");
        goto error_check_header;
    }

    type_line = (const char *)asset_buf;
    resolution_line = &asset_buf[line_end[0] + 1];
    scale_line = &asset_buf[line_end[1] + 1];

    if (strncmp(type_line, "Pf", 2) != 0) {
        fprintf(stderr, "Only support loading single component (Pf) portable float map files\n");
        goto error_check_header;
    }

    width = (int)strtoul(resolution_line, &width_end, 10);
    if (*width_end != ' ') {
        fprintf(stderr, "Expected <space> after PFM resoltion width\n");
        goto error_check_header;
    }

    height = (int)strtoul(width_end  + 1, &height_end, 10);
    if (height_end != (asset_buf + line_end[1])) {
        fprintf(stderr, "Nothing expected after PFM resoltion height\n");
        goto error_check_header;
    }

    if (width == 0 || width > 8192 || height == 0 || height > 8192) {
        fprintf(stderr, "Invalid PFM image width / height (%dx%d)\n", width, height);
        goto error_check_header;
    }

    scale = strtod(scale_line, &scale_end);
    if (scale_end != (asset_buf + line_end[2])) {
        fprintf(stderr, "Nothing expected after PFM scale");
        goto error_check_header;
    }
    if (scale >= 0) {
        fprintf(stderr, "Only support loading little endian PFM files\n");
        goto error_check_header;
    }

    pfm_data_start = scale_end + 1;
    pfm_data_len = sizeof(float) * width * height;

    ret = (struct image *)xmalloc(sizeof(struct image) + pfm_data_len);
    ret->format = IMAGE_FORMAT_XFLOAT;
    ret->stride = sizeof(float) * width;
    ret->width = width;
    ret->height = height;
    ret->data_float = (float *)(ret + 1);

    memcpy(ret->data_float, pfm_data_start, pfm_data_len);

    debug("read PFM %s (%dx%d) OK\n", filename, width, height);

error_check_header:
    munmap((void *)asset_buf, asset_buf_len);
error_mmap:
    close(fd);

    return ret;
}

static bool
write_exr(const char *filename,
          struct image *image,
          enum image_format format)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    exr_image.num_channels = 1;
    exr_image.width = image->width;
    exr_image.height = image->height;

    unsigned char *image_ptr = (unsigned char *)image->data_u8;
    exr_image.images = &image_ptr;

    header.num_channels = 1;
    EXRChannelInfo channel_info;
    header.channels = &channel_info;
    strcpy(channel_info.name, "Y");

    int input_format = (image->format == IMAGE_FORMAT_XFLOAT ?
                        TINYEXR_PIXELTYPE_FLOAT : TINYEXR_PIXELTYPE_HALF);
    int final_format = (format == IMAGE_FORMAT_XFLOAT ?
                        TINYEXR_PIXELTYPE_FLOAT : TINYEXR_PIXELTYPE_HALF);
    header.pixel_types = &input_format;
    header.requested_pixel_types = &final_format;

    const char *err = NULL;
    if (SaveEXRImageToFile(&exr_image, &header, filename, &err) != TINYEXR_SUCCESS) {
        fprintf(stderr, "Failed to save EXR: %s\n", err);
        return false;
    } else
        return true;
}

static void
usage(void)
{
    printf("Usage pfm-to-exr [options] <in_pfm_file> <out_exr_file>\n"
           "\n"
           "    --half              Write a half-float channel (otherwise\n"
           "                        writes full-float)\n"
           "\n"
           "    -h,--help           Display this help\n\n"
           "\n");
    exit(1);
}

int
main(int argc, char **argv)
{
    int opt;

#define HALF_OPT    (CHAR_MAX + 1) // no short opt

    /* N.B. The initial '+' means that getopt will stop looking for options
     * after the first non-option argument...
     */
    const char *short_options="+h";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"half",            no_argument,        0, HALF_OPT},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
            case 'h':
                usage();
                return 0;
            case HALF_OPT:
                write_half_float = true;
                break;
        }
    }

    if (optind != argc - 2)
        usage();

    struct image *img = load_pfm(argv[optind]);

    if (write_half_float)
        write_exr(argv[optind + 1], img, IMAGE_FORMAT_XHALF);
    else
        write_exr(argv[optind + 1], img, IMAGE_FORMAT_XFLOAT);
    return 0;
}
