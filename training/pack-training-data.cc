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
#include <sys/epoll.h>

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
#include <signal.h>

#include <type_traits>
#include <queue>
#include <random>
#include <atomic>

#include "tinyexr.h"

#include "half.hpp"

#include "image_utils.h"
#include "llist.h"
#include "xalloc.h"
#include "pack.h"

#ifdef DEBUG
#define PNG_DEBUG 3
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

#include <png.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

#define BACKGROUND_ID 33

#ifndef offsetof
#define offsetof(s_name, n_name) (size_t)(char *)&(((s_name *)0)->m_name)
#endif


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


/* Frame renders are grouped by directories where the clothes are the same
 */
struct mocap_section {
    char *dir;
    std::vector<char *> files;
};


/* Tracks the asynchronous loading of all the component files for a single
 * frame.
 */
struct pending_frame;

struct pending_file {
    struct pending_frame *frame;
    char filename[1024];

    /* handled by the read thread before handing over to the decode threads */
    int fd;
    uint8_t *buf;
    int len;
    int pos;

    /* handled by the decode thread */
    uint32_t compressed_size;
    char *compressed_data;
};

enum {
    FILE_DEPTH,
    FILE_LABELS,
    FILE_JSON,
    N_FRAME_FILES
};

struct pending_frame {
    struct mocap_section mocap_section;
    int file_idx;

    struct pending_file files[N_FRAME_FILES];

    uint64_t mtime;

    /* handled by the decode thread */
    uint32_t compressed_header_size;
    char *compressed_header;

    /* for writer thread */
    struct pack_frame *pack_frame;
};

struct thread_state {
    pthread_t thread;
    struct pack_file *pack;
};


static const char *top_src_dir;
static const char *out_file;

static bool append_mode = false;
static bool write_half_float = true;
static int n_threads_override = 0;

static pthread_mutex_t read_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct mocap_section> read_queue;
static std::atomic<bool> reading;

static pthread_mutex_t decode_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct pending_frame *> decode_queue;
static pthread_cond_t decode_append_cond = PTHREAD_COND_INITIALIZER;
static std::atomic<bool> decoding;

static pthread_mutex_t write_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct pending_frame *> write_queue;
static pthread_cond_t write_append_cond = PTHREAD_COND_INITIALIZER;

static int indent = 0;

static pthread_once_t cpu_count_once = PTHREAD_ONCE_INIT;
static int n_cpus = 0;

static uint64_t read_io_start;
static std::atomic<std::uint64_t> read_io_frames;
static std::atomic<std::uint64_t> read_io_bytes;

static std::atomic<std::uint64_t> decoded_frames;
static std::atomic<std::uint64_t> compressed_bytes; /* (input to compression) */

static uint64_t write_io_start;
static std::atomic<std::uint64_t> write_io_frames;
static std::atomic<std::uint64_t> write_io_bytes;

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

static void
free_image(struct image *image)
{
    free(image->data_u8);
    free(image);
}

static struct image *
decode_png(uint8_t *buf, int len)
{
    IUImageSpec spec = { 0, 0, IU_FORMAT_U8 };
    if (iu_verify_png_from_memory(buf, len, &spec) != SUCCESS) {
        return NULL;
    }
    struct image *img = xalloc_image(IMAGE_FORMAT_X8, spec.width, spec.height);
    if (iu_read_png_from_memory(buf, len, &spec, (void**)&img->data_u8) !=
        SUCCESS) {
        free_image(img);
        return NULL;
    }

    return img;
}


static struct image *
decode_exr(char *buf, int len, enum image_format fmt)
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


static void
directory_recurse(const char *rel_path)
{
    char label_src_path[1024];
    //char depth_src_path[1024];

    struct stat st;
    DIR *label_dir;
    struct dirent *label_entry;
    char *ext;

    struct mocap_section *mocap_section = NULL;

    xsnprintf(label_src_path, "%s/labels/%s", top_src_dir, rel_path);
    //xsnprintf(depth_src_path, "%s/depth/%s", top_src_dir, rel_path);

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

            if (!mocap_section) {
                struct mocap_section empty;

                read_queue.push(empty);
                mocap_section = &read_queue.back();

                mocap_section->dir = strdup(rel_path);
                mocap_section->files = std::vector<char *>();
            }

            mocap_section->files.push_back(strdup(label_entry->d_name));
        }
    }

    closedir(label_dir);
}

static int
try_to_queue_n_frame_reads(int epollfd, int n_frames)
{
    int n = 0;

    debug("Checking to open more fds\n");
    while (n < n_frames) {
        struct mocap_section mocap_section;

        pthread_mutex_lock(&read_queue_lock);
        if (!read_queue.empty()) {
            mocap_section = read_queue.front();
            read_queue.pop();
        } else {
            pthread_mutex_unlock(&read_queue_lock);
            return n;
        }
        pthread_mutex_unlock(&read_queue_lock);

        for (unsigned i = 0; i < mocap_section.files.size(); i++) {
            debug("Read thread processing %s/%s\n", mocap_section.dir, mocap_section.files[i]);

            struct pending_frame *frame =
                (struct pending_frame *)xmalloc(sizeof(*frame));

            memset(frame, 0, sizeof(*frame));
            for (unsigned j = 0; j < ARRAY_LEN(frame->files); j++)
                frame->files[j].fd = -1;

            frame->mocap_section = mocap_section;

            xsnprintf(frame->files[FILE_DEPTH].filename, "%s/depth/%s/%.*s.exr",
                      top_src_dir,
                      mocap_section.dir,
                      (int)strlen(mocap_section.files[i]) - 4,
                      mocap_section.files[i]);
            xsnprintf(frame->files[FILE_LABELS].filename, "%s/labels/%s/%.*s.png",
                      top_src_dir,
                      mocap_section.dir,
                      (int)strlen(mocap_section.files[i]) - 4,
                      mocap_section.files[i]);
            xsnprintf(frame->files[FILE_JSON].filename, "%s/labels/%s/%.*s.json",
                      top_src_dir,
                      mocap_section.dir,
                      (int)strlen(mocap_section.files[i]) - 4,
                      mocap_section.files[i]);


            unsigned j;
            for (j = 0; j < ARRAY_LEN(frame->files); j++) {
                struct pending_file *file = frame->files + j;
                struct stat st;
                struct epoll_event ev;

                file->frame = frame;

                if (stat(file->filename, &st) < 0) {
                    fprintf(stderr, "SKIP FRAME: Failed to stat %s: %m\n", file->filename);
                    break;
                }

                frame->mtime = st.st_mtim.tv_sec;

                file->len = st.st_size;
                file->buf = (uint8_t *)xmalloc(st.st_size);

                file->fd = open(file->filename, O_RDONLY|O_NONBLOCK|O_CLOEXEC);
                if (file->fd < 0) {
                    fprintf(stderr, "SKIP FRAME: Failed to open depth buffer file %s: %m\n", file->filename);
                    continue;
                }

                ev.events = EPOLLIN;
                ev.data.ptr = file;
                if (epoll_ctl(epollfd, EPOLL_CTL_ADD, file->fd, &ev) < 0) {
                    fprintf(stderr, "Failed to add fd (%d) for file %s to epoll fd (%d): %m\n",
                            file->fd, file->filename, epollfd);
                }
            }
            if (j != ARRAY_LEN(frame->files)) {
                for (j--; j >= 0; j--) {
                    close(frame->files[j].fd);
                    free(frame->files[j].buf);
                }
                free(frame);
                continue;
            }

            n++;
        }
    }

    return n;
}

static int
read_until_eagain(struct pending_file *file)
{
    int n_bytes = 0;
    int rem;

    while ((rem = file->len - file->pos)) {
        int ret = read(file->fd, file->buf + file->pos, rem);
        if (ret == -1) {
            if (errno == EAGAIN)
                return n_bytes;
            else if (errno == EINTR)
                continue;
            else {
                fprintf(stderr, "IO error reading %s: %m\n", file->filename);
                exit(1);
            }
        } else if (ret == 0) {
            /* We only read in response to a POLLIN and we also pre-determine
             * the length of the file and request to read up to that length
             * so shouldn't ever reach EOF
             */
            fprintf(stderr, "Spurious EOF reading %s, with remainder = %d\n", file->filename, rem);
            exit(1);
        }

        debug("read some of %s\n", file->filename);
        file->pos += ret;
        n_bytes += ret;
    }

    /* We only read in response to a POLLIN so don't expect to see an EAGAIN
     * before reading something
     */
    assert(n_bytes);

    return n_bytes;
}

static const char *
get_bandwidth_units_suffix(uint64_t duration_ns, uint64_t byte_count)
{
    double elapsed_sec = duration_ns / 1e9;
    double units_per_second = (double)byte_count / elapsed_sec;

    if (units_per_second > 1024 * 1024 * 1024)
        return "GiB/s";
    else if (units_per_second > 1024 * 1024)
        return "MiB/s";
    else if (units_per_second > 1024)
        return "KiB/s";
    else
        return "B/s";
}

static double
get_bandwidth(uint64_t duration_ns, uint64_t byte_count)
{
    double elapsed_sec = duration_ns / 1e9;
    double units_per_second = (double)byte_count / elapsed_sec;

    while (units_per_second > 1024)
        units_per_second /= 1024;

    return units_per_second;
}

static void *
read_io_thread_cb(void *data)
{
    debug("Running read IO thread\n");

    int epollfd = epoll_create1(0);
    if (epollfd < 0) {
        fprintf(stderr, "Failed to create an epoll file descriptor: %m\n");
        exit(1);
    }

    int max_pending_frames = 333;
    int n_pending_frames = 0;

    uint64_t last_timestamp = get_time();
    uint64_t thread_bytes_read = 0;
    uint64_t last_io_bytes = 0;

    while (1)
    {
        if (n_pending_frames < max_pending_frames) {
            int request = max_pending_frames - n_pending_frames;

            debug("Trying to queue more %d more frame reads (n_pending_frames = %d)\n",
                  request, n_pending_frames);
            n_pending_frames += try_to_queue_n_frame_reads(epollfd, request);
        }

        if (n_pending_frames == 0) {
            bool is_empty;

            debug("Read thread has no pending reads after attempt to queue\n");

            pthread_mutex_lock(&read_queue_lock);
            is_empty = read_queue.empty();
            pthread_mutex_unlock(&read_queue_lock);

            if (is_empty) {
                debug("Thread confirmed read queue is empty\n");
                break;
            }
        }

        int max_events = n_pending_frames * N_FRAME_FILES;
        struct epoll_event events[max_events];

        int n_ev;
        while ((n_ev = epoll_wait(epollfd, events, max_events, -1)) < 0 &&
                errno == SIGINT)
            ;

        if (n_ev < 0) {
            fprintf(stderr, "IO Error: Failed to wait for any epoll events: %m\n");
            exit(1);
        }

        for (int i = 0; i < n_ev; i++) {
            struct pending_file *file = (struct pending_file *)events[i].data.ptr;

            int n_bytes = read_until_eagain(file);

            read_io_bytes += n_bytes;
            thread_bytes_read += n_bytes;

            if (file->pos == file->len) {
                debug("finished reading %s (closing)\n", file->filename);
                close(file->fd);
                file->fd = -1;

                struct pending_frame *frame = file->frame;

                bool still_pending = false;
                for (int j = 0; j < N_FRAME_FILES; j++) {
                    if (frame->files[j].fd != -1)
                        still_pending = true;
                }
                if (!still_pending) {
                    debug("Read frame, queuing for write: %s\n", frame->files[0].filename);
                    n_pending_frames--;

                    read_io_frames++;

                    pthread_mutex_lock(&decode_queue_lock);
                    decode_queue.push(frame);
                    pthread_cond_signal(&decode_append_cond);
                    pthread_mutex_unlock(&decode_queue_lock);
                }
            }
        }

        uint64_t timestamp = get_time();
        uint64_t elapsed = timestamp - last_timestamp;

        if (elapsed > 1000000000) {
            uint64_t byte_count = thread_bytes_read - last_io_bytes;

            debug("Thread read bandwidth = %.3f%s\n",
                  get_bandwidth(elapsed, byte_count),
                  get_bandwidth_units_suffix(elapsed, byte_count));

            last_timestamp = timestamp;
            last_io_bytes = thread_bytes_read;
        }
    }

    debug("Reading thread finished\n");

    return NULL;
}

static void
free_frame(struct pending_frame *frame)
{
    for (int i = 0; i < N_FRAME_FILES; i++) {
        free(frame->files[i].buf);
        free(frame->files[i].compressed_data);
    }
    free(frame->compressed_header);
    free(frame);
}

static bool
decode_frame(struct thread_state *state,
             struct pending_frame *frame)
{
    struct image *labels = NULL, *depth = NULL;

    char *err = NULL;
    int width, height;

    char *path, *section_path, *mocap_path, *mocap_name, *image_name;
    unsigned frame_no = 0;

    uint32_t frame_file_size;

    bool ret = false;

    char *error = NULL;
    char *ignore0 = NULL, *ignore1 = NULL;


    labels = decode_png(frame->files[FILE_LABELS].buf,
                        frame->files[FILE_LABELS].len);
    if (!labels) {
        fprintf(stderr, "Failed to decode frame labels PNG %s\n",
                frame->files[FILE_LABELS].filename);
        return false;
    }

    err = NULL;
    width = pack_get_i64(state->pack, "width", &err);
    if (err) {
        pack_set_i64(state->pack, "width", labels->width);
        free(err);
    } else if (labels->width != width) {
        fprintf(stderr, "Decoded frame labels width (%d) doesn't match pack width (%d)\n",
                labels->width, width);
        goto done;
    }

    height = pack_get_i64(state->pack, "height", &err);
    if (err) {
        pack_set_i64(state->pack, "height", labels->height);
        free(err);
    } else if (labels->height != height) {
        fprintf(stderr, "Decoded frame labels height (%d) doesn't match pack height (%d)\n",
                labels->height, height);
        goto done;
    }

    enum image_format fmt;
    if (write_half_float)
        fmt = IMAGE_FORMAT_XHALF;
    else
        fmt = IMAGE_FORMAT_XFLOAT;


    depth = decode_exr((char *)frame->files[FILE_DEPTH].buf,
                       frame->files[FILE_DEPTH].len,
                       fmt);
    if (!depth) {
        fprintf(stderr, "Failed to decode frame depth EXR %s\n",
                frame->files[FILE_DEPTH].filename);
        goto done;
    }

    if (depth->width != width) {
        fprintf(stderr, "Decoded frame depth width (%d) doesn't match pack width (%d)\n",
                depth->width, width);
        goto done;
    }

    if (depth->height != height) {
        fprintf(stderr, "Decoded frame depth height (%d) doesn't match pack height (%d)\n",
                depth->height, height);
        goto done;
    }

    frame->pack_frame = pack_frame_new(state->pack);

    pack_frame_set_section(frame->pack_frame, "labels",
                           frame->files[FILE_LABELS].buf,
                           frame->files[FILE_LABELS].len);
    pack_frame_set_section(frame->pack_frame, "depth",
                           frame->files[FILE_DEPTH].buf,
                           frame->files[FILE_DEPTH].len);
    pack_frame_set_section(frame->pack_frame, "meta",
                           frame->files[FILE_JSON].buf,
                           frame->files[FILE_JSON].len);

    pack_frame_set_i64(frame->pack_frame, "mtime", frame->mtime);

    path = strdup(frame->files[FILE_LABELS].filename);
    section_path = dirname(path);
    mocap_path = dirname(section_path);
    mocap_name = basename(mocap_path);
    pack_frame_set_string(frame->pack_frame, "mocap-name", mocap_name);
    free(path);

    path = strdup(frame->files[FILE_LABELS].filename);
    image_name = basename(path);
    sscanf(image_name, "Image%04u.png", &frame_no);
    pack_frame_set_i64(frame->pack_frame, "frame", frame_no);
    free(path);

    compressed_bytes = pack_frame_compress(frame->pack_frame, &error);
    if (compressed_bytes == 0) {
        fprintf(stderr, "Failed to compress frame: %s\n", error);
        goto done;
    }

    frame_file_size = 0;
    for (int i = 0; i < N_FRAME_FILES; i++)
        frame_file_size += frame->files[i].len;

    decoded_frames++;

    debug("decoded %s/%u frame\n",
          pack_frame_get_string(frame->pack_frame, "mocap-name", &ignore0),
          (unsigned)pack_frame_get_i64(frame->pack_frame, "mocap-frame", &ignore1));
    free(ignore0);
    free(ignore1);

    ret = true;

done:
    if (depth)
        free_image(depth);
    if (labels)
        free_image(labels);

    return ret;
}

static void *
decode_io_thread_cb(void *data)
{
    struct thread_state *state = (struct thread_state *)data;

    //int n_frames = 0;
    //uint64_t start = get_time();

    while (1) {
        struct pending_frame *frame = NULL;

        pthread_mutex_lock(&decode_queue_lock);
        if (!decode_queue.empty()) {
            frame = decode_queue.front();
            decode_queue.pop();
        } else {
            if (reading) {
                debug("Decode thread %lu waiting for more data\n",
                       (unsigned long)state->thread);
                pthread_cond_wait(&decode_append_cond, &decode_queue_lock);
                pthread_mutex_unlock(&decode_queue_lock);
                continue;
            } else {
                debug("Decode thread finished\n");
                pthread_mutex_unlock(&decode_queue_lock);
                break;
            }
        }
        pthread_mutex_unlock(&decode_queue_lock);

#if 0
        n_frames++;
        if ((n_frames % 100)  == 0) {
            uint64_t time = get_time();
            uint64_t duration_ns = time - start;

            double fps = (double)n_frames / ((double)duration_ns / 1e9);

            printf("Decode thread %lu has processed 100 frames at %u fps\n",
                   (unsigned long)*thread,
                   (unsigned)fps);
        }
#endif

#if 1
        if (decode_frame(state, frame)) {
            pthread_mutex_lock(&write_queue_lock);
            write_queue.push(frame);
            pthread_cond_signal(&write_append_cond);
            pthread_mutex_unlock(&write_queue_lock);
        } else
            free_frame(frame);
#else
#warning "fixme"
        decode_frame(frame);
        free_frame(frame);
#endif
    }

    return NULL;
}

#if 0
static void
print_header(FILE *fp, struct file_header_part0 *header)
{
    fprintf(fp, "version: %u.%u\n", header->major_version, header->minor_version);
    fprintf(fp, "n sections: %u\n", header->n_sections);
    for (unsigned i = 0; i < header->n_sections; i++)
        fprintf(fp, "> [%u]] = %s\n", i, header->section_names[i]);
    fprintf(fp, "n properties: %u\n", header->n_properties);
}
#endif

static void *
write_io_thread_cb(void *data)
{
    struct thread_state *state = (struct thread_state *)data;

    while (1) {
        struct pending_frame *frame = NULL;

        pthread_mutex_lock(&write_queue_lock);
        if (!write_queue.empty()) {
            frame = write_queue.front();
            write_queue.pop();
        } else {
            if (decoding) {
                debug("Write thread waiting for more data\n");
                pthread_cond_wait(&write_append_cond, &write_queue_lock);
                pthread_mutex_unlock(&write_queue_lock);
                continue;
            } else {
                debug("Write thread finished\n");
                pthread_mutex_unlock(&write_queue_lock);
                break;
            }
        }
        pthread_mutex_unlock(&write_queue_lock);

        debug("Write thread processing %s\n", frame->files[0].filename);
        if (!write_io_start)
            write_io_start = get_time();

        write_io_bytes += pack_append_frame(state->pack, frame->pack_frame);
        write_io_frames++;

        free_frame(frame);
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
"This tool will pack a directory of training data into a single file which\n"
"is optimized for processing the frames as a stream for pre-processing or for\n"
"fast loading while training\n"
"\n"
"The input files are expected to be laid out like:\n"
"  /top/depth/<mocap>/<section>/ImageXXXX.exr\n"
"  /top/labels/<mocap>/<section>/ImageXXXX.png\n"
"  /top/labels/<mocap>/<section>/ImageXXXX.json\n"
"\n"
"\n"
"Usage pack-training-data [options] <top_src> <pack_file>\n"
"\n"
"    -a,--append                Append frames to existing pack file\n"
"    -f,--full                  Store full-float channel depth images (otherwise\n"
"                               stores half-float)\n"
"    -t,--threads=<n>           Override how many decoder threads are run\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n");

    exit(1);
}

int
main(int argc, char **argv)
{
    int opt;

    /* N.B. The initial '+' means that getopt will stop looking for options
     * after the first non-option argument...
     */
    const char *short_options="+haft:";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"append",          no_argument,        0, 'a'},
        {"full",            no_argument,        0, 'f'},
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
            case 'a':
                append_mode = true;
                break;
            case 'f':
                write_half_float = false;
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

    top_src_dir = argv[optind];
    out_file = argv[optind + 1];

    if (!append_mode)
        remove(out_file);

    char *err = NULL;
    struct pack_file *pack = pack_open(out_file, &err);
    if (!pack) {
        fprintf(stderr, "Failed to open pack: %s\n", err);
        exit(1);
    }

    if (append_mode) {
        if (pack->n_sections != 3 ||
            strcmp(pack->section_names[0], "labels") != 0 ||
            strcmp(pack->section_names[1], "depth") != 0 ||
            strcmp(pack->section_names[2], "meta") != 0)
        {
            fprintf(stderr, "Incompatible pack file (mismatching frame size or sections).\n");
            exit(1);
        }
    } else {
        pack_declare_frame_section(pack, "labels");
        pack_declare_frame_section(pack, "depth");
        pack_declare_frame_section(pack, "meta");
    }


    printf("Queuing frames to process...\n");

    uint64_t start = get_time();
    directory_recurse("" /* initially empty relative path */);
    uint64_t end = get_time();

    uint64_t duration_ns = end - start;
    printf("%d directories queued to process, in %.3f%s\n",
           (int)read_queue.size(),
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    int n_threads;

    if (!n_threads_override) {
        int n_cpus = cpu_count();
        n_threads = n_cpus * 2;
    } else {
        n_threads = n_threads_override;
    }

    //n_threads = 1;

    printf("Spawning a read, write and %d decode threads\n", n_threads);

    start = get_time();

    read_io_start = get_time();
    reading = true;

    pthread_t read_thread;
    pthread_create(&read_thread,
                   NULL, //attributes
                   read_io_thread_cb,
                   &read_thread); //data
    pthread_setname_np(read_thread, "reader");

    decoding = true;
    struct thread_state *decode_threads = (struct thread_state *)
        xcalloc(n_threads_override, sizeof(decode_threads[0]));
    for (int i = 0; i < n_threads; i++) {
        decode_threads[i].pack = pack;
        pthread_create(&decode_threads[i].thread,
                       NULL, //attributes
                       decode_io_thread_cb,
                       &decode_threads[i]); //data
        pthread_setname_np(decode_threads[i].thread, "decoder");
    }

    thread_state write_thread;
    write_thread.pack = pack;

    pthread_create(&write_thread.thread,
                   NULL, //attributes
                   write_io_thread_cb,
                   &write_thread); //data
    pthread_setname_np(write_thread.thread, "writer");

    uint64_t read_io_end;

    uint64_t last_read_bytes = 0;
    uint64_t last_read_frames = 0;
    uint64_t last_decode_frames = 0;
    uint64_t last_write_bytes = 0;
    uint64_t last_write_frames = 0;
    uint64_t last_read_check_timestamp = get_time();
    uint64_t last_decode_check_timestamp = get_time();
    uint64_t last_write_check_timestamp = get_time();

    while (1) {
        void *thread_ret;

        if (reading) {
            if (pthread_tryjoin_np(read_thread, &thread_ret) == 0) {
                read_io_end = get_time();

                debug("Reading finished\n");
                reading = false;

                /* wake decoders to be sure they know reading has finished so they
                 * won't wait indefinitely
                 */
                pthread_cond_broadcast(&decode_append_cond);
            } else {
                uint64_t timestamp = get_time();
                uint64_t bytes_read = read_io_bytes;
                uint64_t frames = read_io_frames;

                uint64_t duration_ns = timestamp - last_read_check_timestamp;
                uint64_t byte_count = bytes_read - last_read_bytes;
                uint64_t frame_count = frames - last_read_frames;

                double fps = (double)frame_count / ((double)duration_ns / 1e9);

                pthread_mutex_lock(&read_queue_lock);
                unsigned queue_len = read_queue.size();
                pthread_mutex_unlock(&read_queue_lock);

                printf("Current read bandwidth = %.3f %s (%u fps), with queue length = %u\n",
                       get_bandwidth(duration_ns, byte_count),
                       get_bandwidth_units_suffix(duration_ns, byte_count),
                       (unsigned)fps,
                       queue_len);

                last_read_check_timestamp = timestamp;
                last_read_bytes = bytes_read;
                last_read_frames = frames;
            }
        }

        if (decoding) {
            bool finished = true;

            for (int i = 0; i < n_threads; i++) {
                pthread_t tid = decode_threads[i].thread;
                if (tid && pthread_tryjoin_np(tid, &thread_ret) == 0)
                    decode_threads[i].thread = 0;
                else if (tid)
                    finished = false;
            }
            if (finished) {
                debug("All decoding threads finished\n");
                decoding = false;

                /* wake writer to be sure it knows decoding has finished so it
                 * won't wait indefinitely
                 */
                pthread_cond_broadcast(&write_append_cond);
            } else {
                uint64_t timestamp = get_time();
                uint64_t frames = decoded_frames;

                uint64_t duration_ns = timestamp - last_decode_check_timestamp;
                uint64_t frame_count = frames - last_decode_frames;

                double fps = (double)frame_count / ((double)duration_ns / 1e9);

                pthread_mutex_lock(&decode_queue_lock);
                unsigned queue_len = decode_queue.size();
                pthread_mutex_unlock(&decode_queue_lock);
                printf("Current decode fps = %u, with queue length = %u\n",
                       (unsigned)fps,
                       queue_len);

                last_decode_check_timestamp = timestamp;
                last_decode_frames = frames;
            }
        }

        if (decoding == false && pthread_tryjoin_np(write_thread.thread,
                                                    &thread_ret) == 0)
        {
            break;
        } else {
            uint64_t timestamp = get_time();
            uint64_t bytes_write = write_io_bytes;
            uint64_t frames = write_io_frames;

            uint64_t duration_ns = timestamp - last_write_check_timestamp;
            uint64_t write_byte_delta = bytes_write - last_write_bytes;
            uint64_t frame_count = frames - last_write_frames;

            double fps = (double)frame_count / ((double)duration_ns / 1e9);

            pthread_mutex_lock(&write_queue_lock);
            unsigned queue_len = write_queue.size();
            pthread_mutex_unlock(&write_queue_lock);

            printf("Current write bandwidth = %.3f %s (%u fps), with queue length = %u\n",
                   get_bandwidth(duration_ns, write_byte_delta),
                   get_bandwidth_units_suffix(duration_ns, write_byte_delta),
                   (unsigned)fps,
                   queue_len);

            last_write_check_timestamp = timestamp;
            last_write_bytes = bytes_write;
            last_write_frames = frames;
        }

        sleep(1);
    }

    if (!pack_write_header(pack, &err)) {
        fprintf(stderr, "Failed to write pack file header: %s\n", err);
        free(err);
    }
    pack_close(pack);

    end = get_time();

    duration_ns = read_io_end - read_io_start;
    printf("Finished: wrote %" PRIu64 " frames\n", (uint64_t)write_io_frames);
    printf("Overall read bandwidth was = %.3f %s for %" PRIu64 " bytes\n",
           get_bandwidth(duration_ns, read_io_bytes),
           get_bandwidth_units_suffix(duration_ns, read_io_bytes),
           (uint64_t)read_io_bytes);

    duration_ns = end - write_io_start;
    printf("Overall write bandwidth was = %.3f %s for %" PRIu64 " bytes\n",
           get_bandwidth(duration_ns, write_io_bytes),
           get_bandwidth_units_suffix(duration_ns, write_io_bytes),
           (uint64_t)write_io_bytes);

    printf("Overall in-file:memory:out-file ratios of %f:1:%f\n",
          (double)read_io_bytes / (double)compressed_bytes,
          (double)write_io_bytes / (double)compressed_bytes);

    duration_ns = end - start;
    printf("Total time taken was %.3f%s\n",
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    return 0;
}
