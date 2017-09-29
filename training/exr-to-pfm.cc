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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <dirent.h>
#include <stdint.h>
#include <libgen.h>
#include <fcntl.h>

#include "tinyexr.h"

#include "half.hpp"


#ifdef DEBUG
#define PNG_DEBUG 3
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

using half_float::half;

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

static void *
xmalloc(size_t size)
{
    void *ret = malloc(size);
    if (ret == NULL)
        exit(1);
    return ret;
}

static char *
read_file(const char *filename, int *len)
{
    int fd;
    struct stat st;
    char *buf;
    int n = 0;

    *len = 0;

    fd = open(filename, O_RDONLY|O_CLOEXEC);
    if (fd < 0)
        return NULL;

    if (fstat(fd, &st) < 0)
        return NULL;

    // N.B. st_size may be zero for special files like /proc/ files so we
    // speculate
    if (st.st_size == 0)
        st.st_size = 1024;

    buf = (char *)xmalloc(st.st_size);

    while (n < st.st_size) {
        int ret = read(fd, buf + n, st.st_size - n);
        if (ret == -1) {
            if (errno == EINTR)
                continue;
            else {
                free(buf);
                return NULL;
            }
        } else if (ret == 0)
            break;
        else
            n += ret;
    }

    close(fd);

    *len = n;

    return buf;
}

static struct image *
xalloc_image(enum image_format format,
             int width,
             int height)
{
    struct image *img = (struct image *)xmalloc(sizeof(struct image));
    img->format = format;
    img->width = width;
    img->height = height;

    switch (format) {
    case IMAGE_FORMAT_X8:
        img->stride = width;
        break;
    case IMAGE_FORMAT_XFLOAT:
        img->stride = width * sizeof(float);
        break;
    case IMAGE_FORMAT_XHALF:
        img->stride = width * sizeof(half);
        break;
    }
    img->data_u8 = (uint8_t *)xmalloc(img->stride * img->height);

    return img;
}

static struct image *
decode_exr(uint8_t *buf, int len, enum image_format fmt)
{
    EXRVersion version;
    EXRHeader header;
    const char *err = NULL;
    int ret;
    int channel;
    int pixel_type;

    if (fmt != IMAGE_FORMAT_XHALF && fmt != IMAGE_FORMAT_XFLOAT) {
        fprintf(stderr, "Can only decode EXR into full or half float image\n");
        abort();
    }

    ParseEXRVersionFromMemory(&version, (unsigned char *)buf, len);

    if (version.multipart || version.non_image) {
        fprintf(stderr, "Can't load multipart or DeepImage EXR image\n");
        return NULL;
    }

    ret = ParseEXRHeaderFromMemory(&header, &version, (unsigned char *)buf, len, &err);
    if (ret != 0) {
        fprintf(stderr, "Failed to parse EXR header: %s\n", err);
        return NULL;
    }

    channel = -1;
    for (int i = 0; i < header.num_channels; i++) {
        const char *names[] = { "Y", "R", "G", "B" };
        for (unsigned j = 0; j < ARRAY_LEN(names); j++) {
            if (strcmp(names[j], header.channels[i].name) == 0) {
                channel = i;
                break;
            }
        }
        if (channel > 0)
            break;
    }
    if (channel == -1) {
        fprintf(stderr, "Failed to find R, G, B or Y channel in EXR file\n");
        return NULL;
    }

    pixel_type = header.channels[channel].pixel_type;
    if (pixel_type != TINYEXR_PIXELTYPE_HALF && pixel_type != TINYEXR_PIXELTYPE_FLOAT) {
        fprintf(stderr, "Can only decode EXR images with FLOAT or HALF data\n");
        return NULL;
    }

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    ret = LoadEXRImageFromMemory(&exr_image, &header, (const unsigned char *)buf, len, &err);
    if (ret != 0) {
        fprintf(stderr, "Failed to load EXR file: %s\n", err);
    }

    struct image *img = xalloc_image(fmt, exr_image.width, exr_image.height);

    enum image_format exr_fmt = (pixel_type == TINYEXR_PIXELTYPE_HALF ?
                                 IMAGE_FORMAT_XHALF :
                                 IMAGE_FORMAT_XFLOAT);
    if (exr_fmt == fmt) {
        memcpy(img->data_u8, &exr_image.images[channel][0],
               img->stride * img->height);
    } else {
        /* Need to handle format conversion... */

        if (exr_fmt == IMAGE_FORMAT_XHALF) {
            const half *half_pixels = (half *)(exr_image.images[channel]);

            for (int y = 0; y < exr_image.height; y++) {
                const half *exr_row = half_pixels + y * exr_image.width;
                float *out_row = img->data_float + y * exr_image.width;

                for (int x = 0; x < exr_image.width; x++)
                    out_row[x] = exr_row[x];
            }
        } else {
            const float *float_pixels = (float *)(exr_image.images[channel]);

            for (int y = 0; y < exr_image.height; y++) {
                const float *exr_row = float_pixels + y * exr_image.width;
                half *out_row = img->data_half + y * exr_image.width;

                for (int x = 0; x < exr_image.width; x++)
                    out_row[x] = exr_row[x];
            }
        }
    }

    FreeEXRImage(&exr_image);

    return img;
}

static bool
write_pfm_file(struct image *image, const char *filename)
{
    int fd;
    char header[128];
    size_t header_len;
    int file_len;
    char *buf;

    fd = open(filename, O_RDWR|O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "Failed to open output %s: %m\n", filename);
        return false;
    }

    header_len = snprintf(header, sizeof(header), "Pf\n%u %u\n%f\n",
                          image->width, image->height, -1.0);
    if (header_len >= sizeof(header)) {
        fprintf(stderr, "Failed to describe PFN header for %s\n", filename);
        close(fd);
        return false;
    }

    file_len = header_len + sizeof(float) * image->width * image->height;

    if (ftruncate(fd, file_len) < 0) {
        fprintf(stderr, "Failed to set file (%s) size: %m\n", filename);
        close(fd);
        return false;
    }

    buf = (char *)mmap(NULL, file_len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (buf != (void *)-1) {
        debug("Mapped PFM %p, len = %d\n", buf, file_len);
    } else {
        fprintf(stderr, "Failed to mmap PFN output: %m\n");
        close(fd);
        return false;
    }

    memcpy(buf, header, header_len);
    memcpy(buf + header_len,
           image->data_float,
           sizeof(float) * image->width * image->height);

    munmap(buf, file_len);
    close(fd);

    debug("Wrote %s PFM file OK\n", filename);

    return true;
}

int
main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <in_file> <out_file>\n", argv[0]);
        exit(1);
    }

    int len;
    char *file = read_file(argv[1], &len);
    if (!file) {
        fprintf(stderr, "Failed to read file %s: %m\n", argv[1]);
        exit(1);
    }
    debug("read %s OK\n", argv[1]);

    struct image *img = decode_exr((uint8_t *)file, len, IMAGE_FORMAT_XFLOAT);
    if (!img)
        exit(1);

    debug("decoded %s OK (%dx%d)\n", argv[1], img->width, img->height);

    free(file);
    file = NULL;

    write_pfm_file(img, argv[2]);

    free(img);

    return 0;
}
