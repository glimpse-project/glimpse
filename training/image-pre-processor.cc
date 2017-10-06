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
#include <assert.h>
#include <pthread.h>
#include <getopt.h>

#include <cmath>

#include <type_traits>
#include <queue>
#include <random>

#include "tinyexr.h"

#include "half.hpp"

#include "parson.h"

#ifdef DEBUG
#define PNG_DEBUG 3
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

#include <png.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define BACKGROUND_ID 33

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


/* Work is grouped by directories where the clothes are the same since we want
 * to diff sequential images to discard redundant frames which makes sense
 * for a single worker thread to handle
 */
struct work {
    char *dir;
    std::vector<char *> files;
};

struct worker_state
{
    int idx;
    pthread_t thread;
};


static const char *top_src_dir;
static const char *top_out_dir;

static FILE *index_fp;

static bool write_half_float = true;
static bool write_palettized_pngs = true;
static bool write_pfm_depth = false;
static float depth_variance_mm = 20;
static float background_depth_m = 1000;
static int min_body_size_px = 3000;
static float min_body_change_percent = 0.1f;
static int n_threads_override = 0;

static int labels_width = 0;
static int labels_height = 0;

static std::vector<struct worker_state> workers;

static pthread_mutex_t work_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct work> work_queue;

static int indent = 0;

static int grey_to_id_map[255];

#define MAX_PACKED_INDEX 33
static int left_to_right_map[MAX_PACKED_INDEX + 1];

static pthread_once_t cpu_count_once = PTHREAD_ONCE_INIT;
static int n_cpus = 0;

static std::default_random_engine rand_generator;

static png_color palette[] = {
    { 0xff, 0x5d, 0xaa },
    { 0xd1, 0x15, 0x40 },
    { 0xda, 0x1d, 0x0e },
    { 0xdd, 0x5d, 0x1e },
    { 0x49, 0xa2, 0x24 },
    { 0x29, 0xdc, 0xe3 },
    { 0x02, 0x68, 0xc2 },
    { 0x90, 0x29, 0xf9 },
    { 0xff, 0x00, 0xcf },
    { 0xef, 0xd2, 0x37 },
    { 0x92, 0xa1, 0x3a },
    { 0x48, 0x21, 0xeb },
    { 0x2f, 0x93, 0xe5 },
    { 0x1d, 0x6b, 0x0e },
    { 0x07, 0x66, 0x4b },
    { 0xfc, 0xaa, 0x98 },
    { 0xb6, 0x85, 0x91 },
    { 0xab, 0xae, 0xf1 },
    { 0x5c, 0x62, 0xe0 },
    { 0x48, 0xf7, 0x36 },
    { 0xa3, 0x63, 0x0d },
    { 0x78, 0x1d, 0x07 },
    { 0x5e, 0x3c, 0x00 },
    { 0x9f, 0x9f, 0x60 },
    { 0x51, 0x76, 0x44 },
    { 0xd4, 0x6d, 0x46 },
    { 0xff, 0xfb, 0x7e },
    { 0xd8, 0x4b, 0x4b },
    { 0xa9, 0x02, 0x52 },
    { 0x0f, 0xc1, 0x66 },
    { 0x2b, 0x5e, 0x44 },
    { 0x00, 0x9c, 0xad },
    { 0x00, 0x40, 0xad },
    { 0x21, 0x21, 0x21 },
};

#define xsnprintf(dest, fmt, ...) do { \
        if (snprintf(dest, sizeof(dest), fmt,  __VA_ARGS__) >= (int)sizeof(dest)) \
            exit(1); \
    } while(0)

static void *
xmalloc(size_t size)
{
    void *ret = malloc(size);
    if (ret == NULL)
        exit(1);
    return ret;
}

static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

const char *
get_duration_ns_print_scale_suffix(uint64_t duration_ns)
{
    if (duration_ns > 1000000000)
        return "s";
    else if (duration_ns > 1000000)
        return "ms";
    else if (duration_ns > 1000)
        return "us";
    else
        return "ns";
}

float
get_duration_ns_print_scale(uint64_t duration_ns)
{
    if (duration_ns > 1000000000)
        return duration_ns / 1e9;
    else if (duration_ns > 1000000)
        return duration_ns / 1e6;
    else if (duration_ns > 1000)
        return duration_ns / 1e3;
    else
        return duration_ns;
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

static uint8_t *
read_file(const char *filename, int *len)
{
    int fd;
    struct stat st;
    uint8_t *buf;
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

    buf = (uint8_t *)xmalloc(st.st_size);

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

static bool
write_file(const char *filename, uint8_t *buf, int len)
{
    int fd = open(filename, O_WRONLY|O_CREAT|O_CLOEXEC, 0644);
    if (fd < 0)
        return false;

    int n = 0;
    while (n < len) {
        int ret = write(fd, buf + n, len - n);
        if (ret == -1) {
            if (errno == EINTR)
                continue;
            else
                return false;
        } else
            n += ret;
    }

    close(fd);

    return true;
}

static void
free_image(struct image *image)
{
    free(image->data_u8);
    free(image);
}

static bool
write_png_file(const char *filename,
                int width, int height,
                png_bytep *row_pointers,
                png_byte color_type,
                png_byte bit_depth)
{
    png_structp png_ptr;
    png_infop info_ptr;
    bool ret = false;

    /* create file */
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return false;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "png_create_write_struct faile\nd");
        goto error_create_write;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "png_create_info_struct failed");
        goto error_create_info;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "PNG write failure");
        goto error_write;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_PLTE(png_ptr, info_ptr, palette, ARRAY_LEN(palette));

    png_write_info(png_ptr, info_ptr);

    png_write_image(png_ptr, row_pointers);

    png_write_end(png_ptr, NULL);

    ret = true;

error_write:
    png_destroy_info_struct(png_ptr, &info_ptr);
error_create_info:
    png_destroy_write_struct(&png_ptr, NULL);
error_create_write:
    fclose(fp);

    return ret;
}

/* Using EXR is a nightmare. If we try and only add an 'R' channel then
 * e.g. Krita will be able to open the file and it looks reasonable,
 * but OpenCV will end up creating an image with the G and B containing
 * uninitialized garbage. If instead we create a 'Y' only image then
 * OpenCV has special handling for that case and loads it as a greyscale
 * image but Krita will bork and warn that it's not supported. We choose
 * the version that works with OpenCV...
 */
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

static bool
write_pfm(struct image *image, const char *filename)
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

static struct image *
load_frame_labels(const char *dir,
                  const char *filename)
{
    char input_filename[1024];

    FILE *fp;

    unsigned char header[8]; // 8 is the maximum size that can be checked
    png_structp png_ptr;
    png_infop info_ptr;

    int width, height;

    png_bytep *rows;

    int row_stride;

    struct image *img = NULL, *ret = NULL;

    xsnprintf(input_filename, "%s/labels/%s/%s", top_src_dir, dir, filename);

    /* open file and test for it being a png */
    fp = fopen(input_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for reading\n", input_filename);
        goto error_open;
    }

    if (fread(header, 1, 8, fp) != 8) {
        fprintf(stderr, "IO error reading %s file\n", input_filename);
        goto error_check_header;
    }
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "%s was not recognised as a PNG file\n", input_filename);
        goto error_check_header;
    }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "png_create_read_struct failed\n");
        goto error_create_read;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "png_create_info_struct failed\n");
        goto error_create_info;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "libpng setjmp failure\n");
        goto error_png_setjmp;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);

    if (labels_width) {
        if (width != labels_width || height != labels_height) {
            fprintf(stderr, "Inconsistent size for %s (%dx%d) of label image (expected %dx%d)\n",
                    input_filename, width, height, labels_width, labels_height);
            exit(1);
        }
    }

    png_read_update_info(png_ptr, info_ptr);

    row_stride = png_get_rowbytes(png_ptr, info_ptr);

    img = xalloc_image(IMAGE_FORMAT_X8, width, height);
    rows = (png_bytep *)alloca(sizeof(png_bytep) * height);

    for (int y = 0; y < height; y++)
        rows[y] = (png_byte *)(img->data_u8 + row_stride * y);

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "png_read_image_error\n");
        goto error_read_image;
    }

    png_read_image(png_ptr, rows);
    debug("read %s/%s (%dx%d) OK\n", dir, filename, width, height);

    for (int y = 0; y < height; y++) {
        uint8_t *row = img->data_u8 + row_stride * y;

        for (int x = 0; x < width; x++) {
            row[x] = grey_to_id_map[row[x]];

            if (row[x] > MAX_PACKED_INDEX) {
                fprintf(stderr, "Failed to map a label value of 0x%x/%d in image %s\n",
                        row[x], row[x],
                        input_filename);
                goto error_read_image;
            }
        }
    }

    ret = img;

error_read_image:
    if (img && ret == NULL)
        free_image(img);
error_png_setjmp:
    png_destroy_info_struct(png_ptr, &info_ptr);
error_create_info:
    png_destroy_read_struct(&png_ptr, NULL, NULL);
error_create_read:

error_check_header:
    fclose(fp);

error_open:

    return ret;
}


static void
flip_frame_depth(struct image *__restrict__ depth,
                 struct image *__restrict__ out)
{
    int width = depth->width;
    int height = depth->height;

    for (int y = 0; y < height; y++) {
        float *depth_row = depth->data_float + y * width;
        float *out_row = out->data_float + y * width;

        for (int x = 0; x < width; x++) {
            int opposite = width - 1 - x;

            out_row[x] = depth_row[opposite];
            out_row[opposite] = depth_row[x];
        }
    }
}

static void
flip_frame_labels(struct image *__restrict__ labels,
                  struct image *__restrict__ out)
{
    int width = labels->width;
    int height = labels->height;

    for (int y = 0; y < height; y++) {
        uint8_t *label_row = labels->data_u8 + y * width;
        uint8_t *out_row = out->data_u8 + y * width;

        for (int x = 0; x < width; x++) {
            int opposite = width - 1 - x;

            out_row[x] = left_to_right_map[label_row[opposite]];
            out_row[opposite] = left_to_right_map[label_row[x]];
        }
    }
}

static bool
frame_diff(struct image *a, struct image *b,
           int *n_different_px_out,
           int *n_body_px_out)
{
    int width = a->width;
    int height = a->height;
    int n_body_px = 0;
    int n_different_px = 0;

    for (int y = 0; y < height; y++) {
        uint8_t *row = a->data_u8 + a->stride * y;

        for (int x = 0; x < width; x++) {
            if (row[x] != BACKGROUND_ID)
                n_body_px++;
        }
    }

    for (int y = 0; y < height; y++) {
        uint8_t *a_row = a->data_u8 + a->stride * y;
        uint8_t *b_row = b->data_u8 + b->stride * y;

        for (int x = 0; x < width; x++) {
            if (a_row[x] != b_row[x])
                n_different_px++;
        }
    }

    *n_different_px_out = n_different_px;
    *n_body_px_out = n_body_px;

    float percent = ((float)n_different_px * 100.0f) / (float)n_body_px;
    if (percent < min_body_change_percent)
        return false;
    else
        return true;
}

static void
frame_add_noise(const struct image *__restrict__ labels,
                const struct image *__restrict__ depth,
                struct image *__restrict__ noisy_labels,
                struct image *__restrict__ noisy_depth)
{
    int width = labels->width;
    int height = labels->height;
    const float *in_depth_px = depth->data_float;
    const uint8_t *in_labels_px = labels->data_u8;
    float *out_depth_px = noisy_depth->data_float;
    uint8_t *out_labels_px = noisy_labels->data_u8;

    rand_generator.seed(234987);

    /* For picking one of 8 random neighbours for fuzzing the silhouettes */
    std::uniform_int_distribution<int> uniform_distribution(0, 7);

    struct rel_pos {
        int x, y;
    } neighbour_position[] = {
        - 1, - 1,
          0, - 1,
          1, - 1,
        - 1,   0,
          1,   0,
        - 1,   1,
          0,   1,
          1,   1,
    };

#define in_depth_at(x, y) *(in_depth_px + width * y + x)
#define in_label_at(x, y) *(in_labels_px + width * y + x)
#define out_depth_at(x, y) *(out_depth_px + width * y + x)
#define out_label_at(x, y) *(out_labels_px + width * y + x)

    memcpy(noisy_labels->data_u8, labels->data_u8, labels->stride);
    memcpy(noisy_depth->data_float, depth->data_float, depth->stride);

    for (int y = 1; y < height - 1; y++) {

        out_label_at(0, y) = in_label_at(0, y);
        out_depth_at(0, y) = in_depth_at(0, y);

        for (int x = 1; x < width - 1; x++) {

            if (in_label_at(x, y) != BACKGROUND_ID) {
                bool edge = false;
                uint8_t neighbour_label[8] = {
                    in_label_at(x - 1, y - 1),
                    in_label_at(x,     y - 1),
                    in_label_at(x + 1, y - 1),
                    in_label_at(x - 1, y),
                    in_label_at(x + 1, y),
                    in_label_at(x - 1, y + 1),
                    in_label_at(x,     y + 1),
                    in_label_at(x + 1, y + 1),
                };

                for (int i = 0; i < 8; i++) {
                    if (neighbour_label[i] == BACKGROUND_ID) {
                        edge = true;
                        break;
                    }
                }

                if (edge) {
                    int neighbour = uniform_distribution(rand_generator);
                    out_label_at(x, y) = neighbour_label[neighbour];

                    struct rel_pos *rel_pos = &neighbour_position[neighbour];
                    out_depth_at(x, y) = in_depth_at(x + rel_pos->x, y + rel_pos->y);
                } else {
                    out_label_at(x, y) = in_label_at(x, y);
                    out_depth_at(x, y) = in_depth_at(x, y);
                }
            } else {
                out_label_at(x, y) = in_label_at(x, y);
                out_depth_at(x, y) = in_depth_at(x, y);
            }

        }

        out_label_at(width - 1, y) = in_label_at(width - 1, y);
        out_depth_at(width - 1, y) = in_depth_at(width - 1, y);
    }

    memcpy(noisy_labels->data_u8 + (height - 1) * width,
           labels->data_u8 + (height - 1) * width,
           labels->stride);
    memcpy(noisy_depth->data_float + (height - 1) * width,
           depth->data_float + (height - 1) * width,
           depth->stride);

    if (depth_variance_mm) {
        /* We use a Gaussian distribution of error offsets for the depth
         * values.
         *
         * According to Wikipedia the full width at tenth of maximum of a
         * Gaussian curve = approximately 4.29193c (where c is the standard
         * deviation which we need to pass to construct this distribution)
         */
        std::normal_distribution<float> gaus_distribution(0, depth_variance_mm / 4.29193f);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (in_label_at(x, y) != BACKGROUND_ID) {
                    float delta_mm = gaus_distribution(rand_generator);
                    out_depth_at(x, y) += (delta_mm / 1000.0f);
                } else {
                    out_depth_at(x, y) = background_depth_m;
                }

                /* just a paranoid sanity check that we aren't */
                if (std::isinf(out_depth_at(x, y)) ||
                    std::isnan(out_depth_at(x, y))) {
                    fprintf(stderr, "Invalid INF value in depth image");
                    exit(1);
                }
            }
        }
    }
#undef in_depth_at
#undef in_label_at
#undef out_depth_at
#undef out_label_at
}


static void
save_frame_depth(const char *dir, const char *filename,
                 struct image *depth)
{
    char output_filename[1024];
    struct stat st;

    xsnprintf(output_filename, "%s/depth/%s/%s", top_out_dir, dir, filename);

    if (write_pfm_depth) {
        char pfm_filename[1024];

        xsnprintf(pfm_filename, "%.*s.pfm",
                  (int)strlen(output_filename) - 4,
                  output_filename);

        if (stat(pfm_filename, &st) != -1) {
            fprintf(stderr, "Skipping PFM file %s as output already exist\n",
                    pfm_filename);
            return;
        }

        write_pfm(depth, pfm_filename);

        debug("wrote %s\n", pfm_filename);
    } else {
        if (stat(output_filename, &st) != -1) {
            fprintf(stderr, "Skipping EXR file %s as output already exist\n",
                    output_filename);
            return;
        }

        if (write_half_float)
            write_exr(output_filename, depth, IMAGE_FORMAT_XHALF);
        else
            write_exr(output_filename, depth, IMAGE_FORMAT_XFLOAT);

        debug("wrote %s\n", output_filename);
    }
}

static bool
save_frame_labels(const char *dir, const char *filename,
                  struct image *labels)
{
    int width = labels->width;
    int height = labels->height;
    int row_stride = labels->stride;
    char output_filename[1024];
    png_bytep rows[height];

    xsnprintf(output_filename, "%s/labels/%s/%s", top_out_dir, dir, filename);

    for (int y = 0; y < height; y++)
        rows[y] = (png_byte *)(labels->data_u8 + row_stride * y);

    struct stat st;

    if (stat(output_filename, &st) == -1) {
        if (!write_png_file(output_filename,
                            width, height,
                            rows,
                            write_palettized_pngs ? PNG_COLOR_TYPE_PALETTE : PNG_COLOR_TYPE_GRAY,
                            8)) { /* bit depth */
            return false;
        }
        debug("wrote %s\n", output_filename);
    } else {
        fprintf(stderr, "SKIP: %s file already exists\n",
                output_filename);
        return false;
    }

    return true;
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

static struct image *
load_frame_depth(const char *dir, const char *filename)
{
    char input_filename[1024];

    xsnprintf(input_filename, "%s/depth/%s/%s", top_src_dir, dir, filename);

    int exr_file_len;
    uint8_t *exr_file = read_file(input_filename, &exr_file_len);
    struct image *depth = decode_exr(exr_file, exr_file_len, IMAGE_FORMAT_XFLOAT);

    free(exr_file);

    debug("read %s/%s (%dx%d) OK\n", dir, filename, depth->width, depth->height);

    return depth;
}

static void
ensure_directory(const char *path)
{
    struct stat st;
    int ret;

    char *dirname_copy = strdup(path);
    char *parent = dirname(dirname_copy);

    if (strcmp(parent, ".") != 0 &&
        strcmp(parent, "..") != 0 &&
        strcmp(parent, "/") != 0)
    {
        ensure_directory(parent);
    }

    free(dirname_copy);

    ret = stat(path, &st);
    if (ret == -1) {
        int ret = mkdir(path, 0777);
        if (ret < 0) {
            fprintf(stderr, "Failed to create destination directory %s: %m\n", path);
            exit(1);
        }
    }
}

static void
directory_recurse(const char *rel_path)
{
    char label_src_path[1024];
    //char depth_src_path[1024];
    char label_dst_path[1024];
    char depth_dst_path[1024];

    struct stat st;
    DIR *label_dir;
    struct dirent *label_entry;
    char *ext;

    struct work *work = NULL;

    xsnprintf(label_src_path, "%s/labels/%s", top_src_dir, rel_path);
    //xsnprintf(depth_src_path, "%s/depth/%s", top_src_dir, rel_path);
    xsnprintf(label_dst_path, "%s/labels/%s", top_out_dir, rel_path);
    xsnprintf(depth_dst_path, "%s/depth/%s", top_out_dir, rel_path);

    ensure_directory(label_dst_path);
    ensure_directory(depth_dst_path);

    label_dir = opendir(label_src_path);

    while ((label_entry = readdir(label_dir)) != NULL) {
        char next_rel_path[1024];
        char next_src_label_path[1024];

        if (strcmp(label_entry->d_name, ".") == 0 ||
            strcmp(label_entry->d_name, "..") == 0)
            continue;

        xsnprintf(next_rel_path, "%s/%s", rel_path, label_entry->d_name);
        xsnprintf(next_src_label_path, "%s/labels/%s", top_src_dir, next_rel_path);

        stat(next_src_label_path, &st);
        if (S_ISDIR(st.st_mode)) {
            debug("%*srecursing into %s\n", indent, "", next_rel_path);
            indent += 2;
            directory_recurse(next_rel_path);
            indent -= 2;
        } else if ((ext = strstr(label_entry->d_name, ".png")) && ext[4] == '\0') {

            if (!work) {
                struct work empty;

                work_queue.push(empty);
                work = &work_queue.back();

                work->dir = strdup(rel_path);
                work->files = std::vector<char *>();
            }

            work->files.push_back(strdup(label_entry->d_name));
        }
    }

    closedir(label_dir);
}

static void *
worker_thread_cb(void *data)
{
    struct worker_state *state = (struct worker_state *)data;
    struct image *noisy_labels = NULL, *noisy_depth = NULL;
    struct image *flipped_labels = NULL, *flipped_depth = NULL;
    char filename[1024];

    debug("Running worker thread\n");

    for (;;) {
        struct work work;

        char label_dir_path[1024];

        struct image *prev_frame_labels = NULL;

        pthread_mutex_lock(&work_queue_lock);
        if (!work_queue.empty()) {
            work = work_queue.front();
            work_queue.pop();
        } else {
            pthread_mutex_unlock(&work_queue_lock);
            debug("Worker thread finished\n");
            return NULL;
        }
        pthread_mutex_unlock(&work_queue_lock);

        xsnprintf(label_dir_path, "%s/labels/%s", top_src_dir, work.dir);

        for (unsigned i = 0; i < work.files.size(); i++) {
            debug("Thread %d: processing %s/%s\n", state->idx, work.dir, work.files[i]);

            struct image *labels = load_frame_labels(work.dir, work.files[i]);

            int n_different_px = 0, n_body_px = 0;
            if (prev_frame_labels) {
                bool differ = frame_diff(labels,
                                         prev_frame_labels,
                                         &n_different_px,
                                         &n_body_px);

                if (n_body_px == 0) {
                    fprintf(stderr, "SKIPPING: %s/%s - spurious frame with no body pixels!\n",
                            work.dir, work.files[i]);
                    free_image(labels);
                    continue;
                }

                if (n_body_px < min_body_size_px) {
                    fprintf(stderr, "SKIPPING: %s/%s - frame with less than %d body pixels\n",
                            work.dir, work.files[i], min_body_size_px);
                    free_image(labels);
                    continue;
                }

                if (!differ) {
                    fprintf(stderr, "SKIPPING: %s/%s - too similar to previous frame (only %d out of %d body pixels differ)\n",
                            work.dir, work.files[i],
                            n_different_px,
                            n_body_px);
                    free_image(labels);
                    continue;
                }
            }

            if (prev_frame_labels)
                free_image(prev_frame_labels);
            prev_frame_labels = labels;

            xsnprintf(filename, "%.*s.exr",
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);

            struct image *depth = load_frame_depth(work.dir, filename);

            if (!noisy_labels) {
                int width = labels->width;
                int height = labels->height;

                noisy_labels = xalloc_image(IMAGE_FORMAT_X8, width, height);
                noisy_depth = xalloc_image(IMAGE_FORMAT_XFLOAT, width, height);

                flipped_labels = xalloc_image(IMAGE_FORMAT_X8, width, height);
                flipped_depth = xalloc_image(IMAGE_FORMAT_XFLOAT, width, height);
            }

            frame_add_noise(labels, depth, noisy_labels, noisy_depth);

            save_frame_labels(work.dir, work.files[i], noisy_labels);
            save_frame_depth(work.dir, filename, noisy_depth);

            char index_name[512];
            xsnprintf(index_name, "%s/%.*s\n",
                      work.dir,
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);
            fwrite(index_name, strlen(index_name), 1, index_fp);

            flip_frame_labels(labels, flipped_labels);
            flip_frame_depth(depth, flipped_depth);
            frame_add_noise(flipped_labels, flipped_depth, noisy_labels, noisy_depth);

            xsnprintf(filename, "%.*s-flipped.png",
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);
            save_frame_labels(work.dir, filename, noisy_labels);

            xsnprintf(filename, "%.*s-flipped.exr",
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);
            save_frame_depth(work.dir, filename, noisy_depth);

            xsnprintf(index_name, "%s/%.*s-flipped\n",
                      work.dir,
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);
            fwrite(index_name, strlen(index_name), 1, index_fp);

            // Note: we don't free the labels here because they are preserved
            // for comparing with the next frame
            free(depth);


            /*
             * Copy the frame's .json metadata
             */

            xsnprintf(filename, "%s/labels/%s/%.*s.json",
                      top_src_dir,
                      work.dir,
                      (int)strlen(work.files[i]) - 4,
                      work.files[i]);

            int len = 0;
            uint8_t *json_data = read_file(filename, &len);
            if (!json_data) {
                fprintf(stderr, "WARNING: Failed to read frame's meta data %s: %m\n",
                        filename);
            }

            if (json_data) {
                xsnprintf(filename, "%s/labels/%s/%.*s.json",
                          top_out_dir,
                          work.dir,
                          (int)strlen(work.files[i]) - 4,
                          work.files[i]);
                if (!write_file(filename, json_data, len)) {
                    fprintf(stderr, "WARNING: Failed to copy frame's meta data to %s: %m\n",
                            filename);
                }

                /* For the -flipped frame we have to flip the x position of
                 * the associated bones...
                 */
                JSON_Value *root_value = json_parse_string((char *)json_data);
                JSON_Object *root = json_object(root_value);
                JSON_Array *bones = json_object_get_array(root, "bones");
                int n_bones = json_array_get_count(bones);

                for (int b = 0; b < n_bones; b++) {
                    JSON_Object *bone = json_array_get_object(bones, b);
                    //const char *name = json_object_get_string(bone, "name");
                    json_object_set_string(bone, "debug", "foo");
                    float x;

                    JSON_Array *head = json_object_get_array(bone, "head");
                    x = json_array_get_number(head, 0);
                    json_array_replace_number(head, 0, -x);

                    JSON_Array *tail = json_object_get_array(bone, "tail");
                    x = json_array_get_number(tail, 0);
                    json_array_replace_number(tail, 0, -x);
                }

                /* For consistency... */
                xsnprintf(filename, "%s/labels/%s/%.*s-flipped.json",
                          top_out_dir,
                          work.dir,
                          (int)strlen(work.files[i]) - 4,
                          work.files[i]);
                if (json_serialize_to_file_pretty(root_value, filename) != JSONSuccess) {
                    fprintf(stderr, "WARNING: Failed to serialize flipped frame's json meta data to %s: %m\n",
                            filename);
                }

                json_value_free(root_value);
                free(json_data);
            }
        }

        if (prev_frame_labels) {
            free_image(prev_frame_labels);
            prev_frame_labels = NULL;
        }
    }

    return NULL;
}

static void
cpu_count_once_cb(void)
{
    uint8_t *buf;
    int len;
    unsigned ignore = 0, max_cpu = 0;

    buf = read_file("/sys/devices/system/cpu/present", &len);
    if (!buf) {
        fprintf(stderr, "Failed to read number of CPUs\n");
        return;
    }

    if (sscanf((char *)buf, "%u-%u", &ignore, &max_cpu) != 2) {
        fprintf(stderr, "Failed to parse /sys/devices/system/cpu/present\n");
        free(buf);
        return;
    }

    free(buf);

    n_cpus = max_cpu + 1;
}

static int
cpu_count(void)
{
    pthread_once(&cpu_count_once, cpu_count_once_cb);

    return n_cpus;
}

static void
usage(void)
{
    printf(
"Usage image-pre-processor [options] <top_src> <top_dest>\n"
"\n"
"    -f,--full                  Write full-float channel depth images (otherwise\n"
"                               writes half-float)\n"
"    -g,--grey                  Write greyscale not palletized label PNGs\n"
"    -p,--pfm                   Write depth data as PFM files\n"
"                               (otherwise depth data is written in EXR format)\n"
"    --variance=<mm>            The randomized variance in mm of the final depth\n"
"                               values (%.0fmm by default)\n"
"    -b,--background=<m>        Depth in meters of background pixels\n"
"                               (default = %.0fm)\n"
"    --min-body-size=<px>       Minimum size of body in pixels\n"
"                               (default = %dpx)\n"
"    --min-body-change=<%%>      Minimum percentage of changed body pixels\n"
"                               between sequential frames\n"
"                               (default = %.3f%%)\n"
"    -t,--threads=<n>           Override how many worker threads are run\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n",
    depth_variance_mm,
    background_depth_m,
    min_body_size_px,
    min_body_change_percent);

    exit(1);
}


int
main(int argc, char **argv)
{
    int opt;

#define VAR_OPT                 (CHAR_MAX + 1)
#define MIN_BODY_PX_OPT         (CHAR_MAX + 2)
#define MIN_BODY_CHNG_PC_OPT    (CHAR_MAX + 3)

    /* N.B. The initial '+' means that getopt will stop looking for options
     * after the first non-option argument...
     */
    const char *short_options="+hfgpv:b:t:";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"full",            no_argument,        0, 'f'},
        {"grey",            no_argument,        0, 'g'},
        {"pfm",             no_argument,        0, 'p'},
        {"variance",        required_argument,  0, VAR_OPT},
        {"background",      required_argument,  0, 'b'},
        {"min-body-size",   required_argument,  0, MIN_BODY_PX_OPT},
        {"min-body-change", required_argument,  0, MIN_BODY_CHNG_PC_OPT},
        {"threads",         required_argument,  0, 't'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        char *end;

        switch (opt) {
            case 'h':
                usage();
                return 0;
            case 'f':
                write_half_float = false;
                break;
            case 'g':
                write_palettized_pngs = false;
                break;
            case 'p':
                write_pfm_depth = true;
                break;
            case VAR_OPT:
                depth_variance_mm = strtod(optarg, &end);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            case 'b':
                background_depth_m = strtod(optarg, &end);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            case MIN_BODY_PX_OPT:
                min_body_size_px = strtoul(optarg, &end, 10);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            case MIN_BODY_CHNG_PC_OPT:
                min_body_change_percent = strtod(optarg, &end);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            case 't':
                n_threads_override = strtoul(optarg, &end, 10);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
           default:
                usage();
                break;
        }
    }

    if (optind != argc - 2)
        usage();

    if (write_pfm_depth && write_half_float) {
        fprintf(stderr, "Not possible to write half float data to PFM files\n");
        exit(1);
    }

    grey_to_id_map[0x07] = 0; // head left
    grey_to_id_map[0x0f] = 1; // head right
    grey_to_id_map[0x16] = 2; // head top left
    grey_to_id_map[0x1d] = 3; // head top right
    grey_to_id_map[0x24] = 4; // neck
    grey_to_id_map[0x2c] = 5; // clavicle left
    grey_to_id_map[0x33] = 6; // clavicle right
    grey_to_id_map[0x3a] = 7; // shoulder left
    grey_to_id_map[0x42] = 8; // upper-arm left
    grey_to_id_map[0x49] = 9; // shoulder right
    grey_to_id_map[0x50] = 10; // upper-arm right
    grey_to_id_map[0x57] = 11; // elbow left
    grey_to_id_map[0x5f] = 12; // forearm left
    grey_to_id_map[0x66] = 13; // elbow right
    grey_to_id_map[0x6d] = 14; // forearm right
    grey_to_id_map[0x75] = 15; // left wrist
    grey_to_id_map[0x7c] = 16; // left hand
    grey_to_id_map[0x83] = 17; // right wrist
    grey_to_id_map[0x8a] = 18; // right hand
    grey_to_id_map[0x92] = 19; // left hip
    grey_to_id_map[0x99] = 20; // left thigh
    grey_to_id_map[0xa0] = 21; // right hip
    grey_to_id_map[0xa8] = 22; // right thigh
    grey_to_id_map[0xaf] = 23; // left knee
    grey_to_id_map[0xb6] = 24; // left shin
    grey_to_id_map[0xbd] = 25; // right knee
    grey_to_id_map[0xc5] = 26; // right shin
    grey_to_id_map[0xcc] = 27; // left ankle
    grey_to_id_map[0xd3] = 28; // left toes
    grey_to_id_map[0xdb] = 29; // right ankle
    grey_to_id_map[0xe2] = 30; // right toes
    grey_to_id_map[0xe9] = 31; // left waist
    grey_to_id_map[0xf0] = 32; // right waist

static_assert(BACKGROUND_ID == 33, "");
    grey_to_id_map[0x40] = BACKGROUND_ID;

    // A few paranoid checks...
    static_assert(MAX_PACKED_INDEX == 33, "Only expecting 33 labels");
    static_assert(ARRAY_LEN(left_to_right_map) == (MAX_PACKED_INDEX + 1),
                  "Only expecting to flip 33 packed labels");

    for (unsigned i = 0; i < ARRAY_LEN(left_to_right_map); i++)
        left_to_right_map[i] = i;

#define flip(A, B) do {  \
        uint8_t tmp = left_to_right_map[A]; \
        left_to_right_map[A] = left_to_right_map[B]; \
        left_to_right_map[B] = tmp; \
    } while(0)

    flip(0, 1); //head
    flip(2, 3); // head top
    flip(5, 6); // clavicle
    flip(7, 9); // shoulder
    flip(8, 10); // upper-arm
    flip(11, 13); // elbow
    flip(12, 14); // forearm
    flip(15, 17); // wrist
    flip(16, 18); // hand
    flip(19, 21); // hip
    flip(20, 22); // thigh
    flip(23, 25); // knee
    flip(24, 26); // shin
    flip(27, 29); // ankle
    flip(28, 30); // toes
    flip(31, 32); // waist

#undef flip

    top_src_dir = argv[optind];
    top_out_dir = argv[optind + 1];

    printf("Queuing frames to process...\n");

    uint64_t start = get_time();
    directory_recurse("" /* initially empty relative path */);
    uint64_t end = get_time();

    uint64_t duration_ns = end - start;
    printf("%d directories queued to process, in %.3f%s\n",
           (int)work_queue.size(),
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    char index_filename[512];
    xsnprintf(index_filename, "%s/index", top_out_dir);
    index_fp = fopen(index_filename, "w");

    int n_threads;

    if (!n_threads_override) {
        int n_cpus = cpu_count();
        n_threads = n_cpus * 2;
    } else {
        n_threads = n_threads_override;
    }

    //n_threads = 1;

    printf("Spawning %d worker threads\n", n_threads);

    workers.resize(n_threads, worker_state());

    start = get_time();

    for (int i = 0; i < n_threads; i++) {
        workers[i].idx = i;
        pthread_create(&workers[i].thread,
                       NULL, //sttributes
                       worker_thread_cb,
                       &workers[i]); //data
    }

    for (int i = 0; i < n_threads; i++) {
        void *ret;

        pthread_join(workers[i].thread, &ret);
    }

    end = get_time();
    duration_ns = end - start;

    fclose(index_fp);
    printf("index written\n");

    printf("Finished processing all frames in %.3f%s\n",
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    return 0;
}
