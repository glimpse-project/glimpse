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

#include <ImfInputFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>
#include <ImfChannelList.h>

#include <ImathBox.h>

#ifdef DEBUG
#define PNG_DEBUG 3
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static int indent = 0;

enum image_format {
    IMAGE_FORMAT_XFLOAT,
};

struct image
{
    enum image_format format;
    int width;
    int height;

    union {
        float *data_float;
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

static const char *
exr_pixel_type_name(enum PixelType type)
{
    switch (type) {
    case FLOAT:
        return "FLOAT";
    case HALF:
        return "HALF";
    case UINT:
        return "UINT";
    case NUM_PIXELTYPES: //keep compiler happy
        exit(1);
        return "FAIL";
    }

    return "FAIL";
}

static struct image *
load_exr_file(const char *filename)
{
    InputFile in_file(filename);

    Box2i dw = in_file.header().dataWindow();

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    const Channel *channel;
    const char *channel_name = NULL;

    const ChannelList &channels = in_file.header().channels();
    for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); i++) {
        const Channel &channel = i.channel();
        const char *name = i.name();
        const char *type_name = exr_pixel_type_name(channel.type);

        debug("EXR: Channel '%s': type %s\n", name, type_name);
    }

    channel = channels.findChannel("Y");
    if (channel) {
        channel_name = "Y";
    } else {
        channel = channels.findChannel("R");
        if (channel)
            channel_name = "R";
        else {
            fprintf(stderr, "Failed to find 'Y' or 'R' channel\n");
            return NULL;
        }
    }

    struct image *ret =
        (struct image *)xmalloc(sizeof(*ret) + sizeof(float) * width * height);
    ret->format = IMAGE_FORMAT_XFLOAT;
    ret->width = width;
    ret->height = height;
    ret->data_float = (float *)(ret + 1);

    FrameBuffer frameBuffer;

    if (channel->type == FLOAT) {
        frameBuffer.insert(channel_name,
                           Slice(FLOAT,
                                 (char *)ret->data_float,
                                 sizeof(float), // x stride,
                                 sizeof(float) * width)); // y stride

        in_file.setFrameBuffer(frameBuffer);
        in_file.readPixels(dw.min.y, dw.max.y);
    } else {
        Array2D<half> depth;
        depth.resizeErase(height, width);

        frameBuffer.insert(channel_name,
                           Slice(HALF,
                                 (char *)&depth[0][0],
                                 sizeof(half), // x stride,
                                 sizeof(half) * width)); // y stride

        in_file.setFrameBuffer(frameBuffer);
        in_file.readPixels(dw.min.y, dw.max.y);

        for (int y = 0; y < height; y++) {
            half *exr_row = &depth[y][0];
            float *out_row = ret->data_float + y * width;

            for (int x = 0; x < width; x++) {
                out_row[x] = exr_row[x];
                //printf("%f ", exr_row[x]);
            }
            //printf("\n");
        }
    }

    debug("%*sread %s (%dx%d) OK\n", indent, "", filename, width, height);

    debug("Keeping \"%s\" %s channel\n",
          channel_name, exr_pixel_type_name(channel->type));
    return ret;
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

    struct image *img = load_exr_file(argv[1]);

    write_pfm_file(img, argv[2]);

    free(img);

    return 0;
}
