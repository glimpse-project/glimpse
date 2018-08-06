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
#include <assert.h>
#include <pthread.h>
#include <getopt.h>
#include <inttypes.h>

#include <cmath>

#include <type_traits>
#include <queue>
#include <random>
#include <atomic>

#include "half.hpp"

#include "parson.h"

#include "glimpse_log.h"
#include "glimpse_data.h"
#include "image_utils.h"
#include "rdt_tree.h"


#ifdef DEBUG
#define PNG_DEBUG 3
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

#include <png.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define BACKGROUND_ID 0

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

static int expected_width;
static int expected_height;
static float expected_fov;

static FILE *index_fp;

static bool write_half_float = true;
static bool write_palettized_pngs = true;
static bool write_pfm_depth = false;
static float depth_variance_mm = 20;
static float background_depth_m = 1000;
static int min_body_size_px = 3000;
static float min_body_change_percent = 0.1f;
static int n_threads_override = 0;

static std::vector<struct worker_state> workers;

static pthread_mutex_t work_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct work> work_queue;

static int indent = 0;

static uint8_t grey_to_id_map[256];
static uint8_t left_to_right_map[256];

static pthread_once_t cpu_count_once = PTHREAD_ONCE_INIT;
static int n_cpus = 0;

static std::default_random_engine rand_generator;

static std::atomic<std::uint64_t> input_frame_count;

static uint64_t max_frame_count = UINT64_MAX;
static std::atomic<std::uint64_t> frame_count;
static std::atomic_bool finished;

static png_color palette[] = {
    { 0x21, 0x21, 0x21 },
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
    { 0xff, 0x5d, 0xaa },
};

#define xsnprintf(dest, fmt, ...) do { \
        if (snprintf(dest, sizeof(dest), fmt,  __VA_ARGS__) >= (int)sizeof(dest)) \
            exit(1); \
    } while(0)

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

    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "Failed to stat with file descriptor for %s\n", filename);
        exit(1);
        return NULL;
    }

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
                fprintf(stderr, "Error reading %s: %m\n", filename);
                free(buf);
                close(fd);
                exit(1);
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
                uint8_t* data)
{
    IUImageSpec spec = { width, height, IU_FORMAT_U8 };
    return iu_write_png_to_file(filename, &spec, (void*)data,
                                write_palettized_pngs ? palette : NULL,
                                ARRAY_LEN(palette)) == SUCCESS;
}

static bool
write_exr(const char *filename,
          struct image *image,
          enum image_format format)
{
    IUImageSpec spec = {
        image->width,
        image->height,
        image->format == IMAGE_FORMAT_XFLOAT ? IU_FORMAT_FLOAT : IU_FORMAT_HALF,
    };
    return iu_write_exr_to_file(filename, &spec, (void*)image->data_u8,
                                (format == IMAGE_FORMAT_XFLOAT ?
                                 IU_FORMAT_FLOAT : IU_FORMAT_HALF)) == SUCCESS;
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
    xsnprintf(input_filename, "%s/labels/%s/%s", top_src_dir, dir, filename);

    IUImageSpec spec = { expected_width, expected_height, IU_FORMAT_U8 };
    struct image *img = xalloc_image(IMAGE_FORMAT_X8,
                                     expected_width, expected_height);

    IUReturnCode code = iu_read_png_from_file(input_filename, &spec,
                                              &img->data_u8,
                                              NULL, // palette output
                                              NULL); // palette size
    if (code != SUCCESS) {
        fprintf(stderr, "Failed to read labels PNG: %s\n",
                iu_code_to_string(code));
        exit(1);
    }

    debug("read %s/%s (%dx%d) OK\n", dir, filename, img->width, img->height);

    for (int y = 0; y < expected_height; y++) {
        uint8_t *row = img->data_u8 + expected_width * y;

        for (int x = 0; x < expected_width; x++)
            row[x] = grey_to_id_map[row[x]];
    }

    return img;
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
    char output_filename[1024];

    xsnprintf(output_filename, "%s/labels/%s/%s", top_out_dir, dir, filename);

    struct stat st;

    if (stat(output_filename, &st) == -1) {
        if (!write_png_file(output_filename,
                            labels->width, labels->height,
                            labels->data_u8)) { /* bit depth */
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
load_frame_depth(const char *dir, const char *filename)
{
    char input_filename[1024];
    xsnprintf(input_filename, "%s/depth/%s/%s", top_src_dir, dir, filename);

    IUImageSpec spec = { expected_width, expected_height, IU_FORMAT_FLOAT };
    struct image *depth = xalloc_image(IMAGE_FORMAT_XFLOAT,
                                       expected_width, expected_height);

    IUReturnCode code = iu_read_exr_from_file(input_filename, &spec,
                                              (void **)&depth->data_u8);
    if (code != SUCCESS) {
        fprintf(stderr, "Failed to read EXR from memory: %s\n",
                iu_code_to_string(code));
        exit(1);
    }

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
        if (ret < 0 && errno != EEXIST) {
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

            input_frame_count++;
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
        char label_dst_path[1024];
        char depth_dst_path[1024];

        struct image *prev_frame_labels = NULL;

        bool ensure_dir_done = false;

        pthread_mutex_lock(&work_queue_lock);
        if (!work_queue.empty()) {
            work = work_queue.front();
            work_queue.pop();
        } else {
            pthread_mutex_unlock(&work_queue_lock);
            debug("Worker thread finished\n");
            break;
        }
        pthread_mutex_unlock(&work_queue_lock);

        xsnprintf(label_dir_path, "%s/labels/%s", top_src_dir, work.dir);

        xsnprintf(label_dst_path, "%s/labels/%s", top_out_dir, work.dir);
        xsnprintf(depth_dst_path, "%s/depth/%s", top_out_dir, work.dir);

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

            /* Check the frame count after checking whether we would skip the
             * frame but before we write anything so we can limit writes
             * according to the max_frame_count threshold
             */
            if (frame_count >= max_frame_count) {
                finished = true;
                break;
            }
            frame_count += 2;

            if (prev_frame_labels)
                free_image(prev_frame_labels);
            prev_frame_labels = labels;

            if (!ensure_dir_done) {
                ensure_directory(label_dst_path);
                ensure_directory(depth_dst_path);
                ensure_dir_done = true;
            }

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
            free_image(depth);


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

        if (finished)
            break;
    }

    if (noisy_labels) {
        free_image(noisy_labels);
        free_image(noisy_depth);
    }
    if (flipped_labels) {
        free_image(flipped_labels);
        free_image(flipped_depth);
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
"Usage image-pre-processor [options] <top_src> <top_dest> <label_map.json>\n"
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
"    -j,--threads=<n>           Override how many worker threads are run\n"
"    -m,--max-frames=<n>        Don't pre-process more than this many frames\n"
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

    struct gm_logger *log = gm_logger_new(NULL, NULL);

    const char *short_options="hfgpv:b:j:m:";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"full",            no_argument,        0, 'f'},
        {"grey",            no_argument,        0, 'g'},
        {"pfm",             no_argument,        0, 'p'},
        {"variance",        required_argument,  0, VAR_OPT},
        {"background",      required_argument,  0, 'b'},
        {"min-body-size",   required_argument,  0, MIN_BODY_PX_OPT},
        {"min-body-change", required_argument,  0, MIN_BODY_CHNG_PC_OPT},
        {"threads",         required_argument,  0, 'j'},
        {"max-frames",      required_argument,  0, 'm'},
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
            case 'j':
                n_threads_override = strtoul(optarg, &end, 10);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            case 'm':
                max_frame_count = strtoul(optarg, &end, 10);
                if (*optarg == '\0' || *end != '\0')
                    usage();
                break;
            default:
                usage();
                break;
        }
    }

    if (argc - optind != 3) {
        usage();
    }

    if (write_pfm_depth && write_half_float) {
        fprintf(stderr, "Not possible to write half float data to PFM files\n");
        exit(1);
    }

    top_src_dir = argv[optind];
    top_out_dir = argv[optind + 1];
    char *label_map_file = argv[optind + 2];

    JSON_Value *label_map = gm_data_load_label_map_from_json(log,
                                                             label_map_file,
                                                             grey_to_id_map,
                                                             NULL); // abort on error
    rdt_util_load_flip_map_from_label_map(log, label_map, left_to_right_map,
                                          NULL); // abort on error

    char meta_filename[512];
    xsnprintf(meta_filename, "%s/meta.json", top_src_dir);
    JSON_Value *meta = json_parse_file(meta_filename);
    if (!meta) {
        fprintf(stderr, "Failed to parse top level meta.json\n");
    }

    JSON_Value *cam = json_object_get_value(json_object(meta), "camera");
    expected_width = json_object_get_number(json_object(cam), "width");
    expected_height = json_object_get_number(json_object(cam), "height");
    expected_fov = json_object_get_number(json_object(cam), "vertical_fov");
    printf("Data rendered at %dx%d with fov = %.3f\n",
           expected_width,
           expected_height,
           expected_fov);

    printf("Queuing frames to process...\n");

    uint64_t start = get_time();
    directory_recurse("" /* initially empty relative path */);
    uint64_t end = get_time();

    uint64_t duration_ns = end - start;
    printf("%d directories queued to process, in %.3f%s\n",
           (int)work_queue.size(),
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    ensure_directory(top_out_dir);

    /* We want to add the label names to the output meta.json but it doesn't
     * make sense to keep the input mappings...
     */
    JSON_Array *label_map_array = json_array(label_map);
    for (int i = 0; i < (int)json_array_get_count(label_map_array); i++) {
        JSON_Object *mapping = json_array_get_object(label_map_array, i);
        json_object_remove(mapping, "inputs");
    }
    json_object_set_value(json_object(meta), "labels", label_map);
    json_object_set_number(json_object(meta), "n_labels",
                           json_array_get_count(label_map_array));

    xsnprintf(meta_filename, "%s/meta.json", top_out_dir);
    if (json_serialize_to_file_pretty(meta, meta_filename) != JSONSuccess) {
        fprintf(stderr, "Failed to write %s\n", meta_filename);
        exit(1);
    }
    json_value_free(meta);
    meta = NULL;

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

    while (true) {
        uint64_t target_frame_count;

        bool finished = false;
        pthread_mutex_lock(&work_queue_lock);
        if (work_queue.empty())
            finished = true;
        pthread_mutex_unlock(&work_queue_lock);
        if (finished)
            break;

        if (max_frame_count != UINT64_MAX)
            target_frame_count = max_frame_count;
        else
            target_frame_count = input_frame_count * 2;

        int progress = 100.0 * ((double)frame_count / (double)target_frame_count);
        printf("\nProgress = %3d%%: %10" PRIu64 " / %-10" PRIu64 "\n\n",
               progress, (uint64_t)frame_count, (uint64_t)target_frame_count);

        sleep(1);
    }

    for (int i = 0; i < n_threads; ++i) {
        void *thread_ret;
        pthread_t tid = workers[i].thread;
        if (pthread_join(tid, &thread_ret) != 0) {
            fprintf(stderr, "Error joining thread, continuing...\n");
        }
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
