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

#include <type_traits>
#include <queue>

#include <ImfInputFile.h>
#include <ImfOutputFile.h>
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

#include <png.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

struct work {
    char *src_label_path;
    char *dst_label_path;
    char *src_depth_path;
    char *dst_depth_path;
};

struct worker_state
{
    int idx;
    pthread_t thread;
};

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static std::vector<struct worker_state> workers;

static pthread_mutex_t work_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static std::queue<struct work> work_queue;

static int indent = 0;

static int grey_to_id_map[255];

#define MAX_PACKED_INDEX 33
static int left_to_right_map[MAX_PACKED_INDEX + 1];

static pthread_once_t cpu_count_once = PTHREAD_ONCE_INIT;
static int n_cpus = 0;


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

static bool
process_png_file(const char *filename, const char *out_filename)
{
    int out_filename_len = strlen(out_filename);

    unsigned char header[8]; // 8 is the maximum size that can be checked
    png_structp png_ptr;
    png_infop info_ptr;

    int width, height;

    png_bytep *input_rows;
    png_bytep input_data;
    png_bytep *id_rows;
    png_bytep id_data;
    png_bytep *flipped_rows;
    png_bytep flipped_data;

    int row_stride;

    char *ids_png_filename, *flipped_png_filename;

    struct stat st;
    FILE *fp;

    bool ret = false;

    if (asprintf(&ids_png_filename, "%.*s.png",
                 out_filename_len - 4, out_filename) == -1)
        exit(1);
    if (asprintf(&flipped_png_filename, "%.*s-flipped.png",
                 out_filename_len - 4, out_filename) == -1)
        exit(1);

    if (stat(ids_png_filename, &st) != -1 && stat(flipped_png_filename, &st) != -1) {
        debug("%*sSkipping PNG processing as outputs already exist\n", indent, "");
        ret = true;
        goto error_stat_outputs;
    }

    /* open file and test for it being a png */
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        goto error_open;
    }

    if (fread(header, 1, 8, fp) != 8) {
        fprintf(stderr, "IO error reading %s file\n", filename);
        goto error_check_header;
    }
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "%s was not recognised as a PNG file\n", filename);
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

    png_read_update_info(png_ptr, info_ptr);

    row_stride = png_get_rowbytes(png_ptr, info_ptr);

    input_rows = (png_bytep *)xmalloc(sizeof(png_bytep) * height);
    input_data = (png_bytep)xmalloc(row_stride * height);

    id_rows = (png_bytep *)xmalloc(sizeof(png_bytep) * height);
    id_data = (png_bytep)xmalloc(row_stride * height);

    flipped_rows = (png_bytep *)xmalloc(sizeof(png_bytep) * height);
    flipped_data = (png_bytep)xmalloc(row_stride * height);

    for (int y = 0; y < height; y++) {
        input_rows[y] = (png_byte *)input_data + row_stride * y;
        id_rows[y] = (png_byte *)id_data + row_stride * y;
        flipped_rows[y] = (png_byte *)flipped_data + row_stride * y;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "png_read_image_error\n");
        goto error_read_image;
    }

    png_read_image(png_ptr, input_rows);
    debug("%*sread %s (%dx%d) OK\n", indent, "", filename, width, height);

    for (int y = 0; y < height; y++) {
        uint8_t *input_row = (uint8_t *)input_data + row_stride * y;
        uint8_t *id_row = (uint8_t *)id_data + row_stride * y;
        uint8_t *flipped_row = (uint8_t *)flipped_data + row_stride * y;

        for (int x = 0; x < width; x++) {
            id_row[x] = grey_to_id_map[input_row[x]];
            if (id_row[x] > MAX_PACKED_INDEX) {
                fprintf(stderr, "Failed to map a label value of 0x%x/%d in image %s\n",
                        input_row[x], input_row[x],
                        filename);
                exit(1);
            }
        }

        /* XXX: assuming even width so we don't have to handle an odd
         * center pixel
         */
        for (int x = 0; x < width; x++) {
            int opposite = width - 1 - x;

            flipped_row[x] = left_to_right_map[id_row[opposite]];
            flipped_row[opposite] = left_to_right_map[id_row[x]];
        }
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error writing PNG with packed IDs\n");
        goto error_write_ids_file;
    }

    if (stat(ids_png_filename, &st) == -1) {
        if (!write_png_file(ids_png_filename,
                            width, height,
                            id_rows,
                            PNG_COLOR_TYPE_GRAY,
                            8)) { /* bit depth */
            goto error_write_ids_file;
        }
        debug("%*swrote %s\n", indent, "", ids_png_filename);
    } else {
        fprintf(stderr, "%*sSKIP: %s file already exists\n",
                indent, "", ids_png_filename);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error writing horizontally flipped PNG\n");
        goto error_write_flipped_file;
    }

    if (stat(flipped_png_filename, &st) == -1) {
        if (!write_png_file(flipped_png_filename,
                            width, height,
                            flipped_rows,
                            PNG_COLOR_TYPE_GRAY,
                            8)) { /* bit depth */
            goto error_write_flipped_file;
        }
        debug("%*swrote %s\n", indent, "", flipped_png_filename);
    } else {
        fprintf(stderr, "%*sSKIP: %s file already exists\n",
                indent, "", flipped_png_filename);
    }

    ret = true;

error_write_ids_file:
error_write_flipped_file:
error_read_image:
    free(input_rows);
    free(id_rows);
    free(flipped_rows);
    free(input_data);
    free(id_data);
    free(flipped_data);
error_png_setjmp:
    png_destroy_info_struct(png_ptr, &info_ptr);
error_create_info:
    png_destroy_read_struct(&png_ptr, NULL, NULL);
error_create_read:

error_check_header:
    fclose(fp);

error_open:
error_stat_outputs:
    free(ids_png_filename);
    free(flipped_png_filename);

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
static void
write_exr(Array2D<float> &pixels, const char *filename)
{
    Header header(pixels.width(), pixels.height());
    header.channels().insert ("Y", Channel(FLOAT));

    OutputFile out_file(filename, header);

    FrameBuffer outFrameBuffer;
    outFrameBuffer.insert ("Y",
                           Slice (FLOAT,
                                  (char *)&pixels[0][0],
                                  sizeof(float), // x stride,
                                  sizeof(float) * pixels.width())); // y stride

    out_file.setFrameBuffer(outFrameBuffer);
    out_file.writePixels(pixels.height());
}

static bool
process_exr_file(const char *filename, const char *out_filename)
{
    int out_filename_len = strlen(out_filename);
    char *exr_filename, *exr_flipped_filename;
    struct stat st;

    if (asprintf(&exr_filename, "%.*s.exr",
                 out_filename_len - 4, out_filename) == -1)
        exit(1);
    if (asprintf(&exr_flipped_filename, "%.*s-flipped.exr",
                 out_filename_len - 4, out_filename) == -1)
        exit(1);

    if (stat(exr_filename, &st) != -1 && stat(exr_flipped_filename, &st) != -1) {
        free(exr_filename);
        free(exr_flipped_filename);
        debug("%*sSkipping EXR file as outputs already exist\n", indent, "");
        return true;
    }

    /* Just for posterity and to vent frustration within comments, the
     * RgbaInputFile and Rgba struct that the openexr documentation recommends
     * for reading typical RGBA EXR images is only good for half float
     * components.
     *
     * We noticed this after seeing lots of 'inf' float values due to out of
     * range floats.
     */
    InputFile in_file(filename);
    Array2D<float> pixels;

    Array2D<float> grey;
    Array2D<float> grey_flipped;

    Box2i dw = in_file.header().dataWindow();

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    pixels.resizeErase(height, width);
    grey.resizeErase(height, width);
    grey_flipped.resizeErase(height, width);

    /* We assume the green and blue channels are redundant and arbitrarily
     * just pick the red channel to read...
     */
    FrameBuffer framebuffer;
    framebuffer.insert ("R",
                        Slice (FLOAT,
                               (char *)&pixels[0][0],
                               sizeof(pixels[0][0]), // x stride,
                               sizeof(float) * pixels.width())); // y stride

    in_file.setFrameBuffer(framebuffer);

#if 0 // uncomment to debug / check the channels available
    const ChannelList &channels = in_file.header().channels();
    for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
        const Channel &channel = i.channel();
        const char *name = i.name();

        debug("EXR: Channel '%s': type %s\n", name, channel.type == FLOAT ? "== FLOAT" : "!= FLOAT");
    }
#endif

    in_file.readPixels(dw.min.y, dw.max.y);

    debug("%*sread %s (%dx%d) OK\n", indent, "", filename, width, height);

    for (int y = 0; y < height; y++) {
        float *red_row = &pixels[y][0];
        float *grey_row = &grey[y][0];
        float *grey_flipped_row = &grey_flipped[y][0];

        for (int x = 0; x < width; x++)
            grey_row[x] = red_row[x];

        for (int x = 0; x < width; x++) {
            int opposite = width - 1 - x;

            grey_flipped_row[x] = red_row[opposite];
            grey_flipped_row[opposite] = red_row[x];
        }
    }

    write_exr(grey, exr_filename);
    debug("%*swrote %s\n", indent, "", exr_filename);
    free(exr_filename);

    write_exr(grey_flipped, exr_flipped_filename);
    debug("%*swrote %s\n", indent, "", exr_flipped_filename);
    free(exr_flipped_filename);

    return true;
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

    free(parent);

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
directory_recurse(const char *src_label_dir_path,
                  const char *src_depth_dir_path,
                  const char *dest_label_dir_path,
                  const char *dest_depth_dir_path)
{
    struct stat st;
    DIR *label_dir;
    struct dirent *label_entry;
    char *ext;

    ensure_directory(dest_label_dir_path);
    ensure_directory(dest_depth_dir_path);

    label_dir = opendir(src_label_dir_path);

    while ((label_entry = readdir(label_dir)) != NULL) {
        char *next_src_label_path, *next_src_depth_path;
        char *next_dest_label_path, *next_dest_depth_path;

        if (strcmp(label_entry->d_name, ".") == 0 || strcmp(label_entry->d_name, "..") == 0)
            continue;

        if (asprintf(&next_src_label_path, "%s/%s", src_label_dir_path, label_entry->d_name) == -1)
            exit(1);
        if (asprintf(&next_src_depth_path, "%s/%s", src_depth_dir_path, label_entry->d_name) == -1)
            exit(1);
        if (asprintf(&next_dest_label_path, "%s/%s", dest_label_dir_path, label_entry->d_name) == -1)
            exit(1);
        if (asprintf(&next_dest_depth_path, "%s/%s", dest_depth_dir_path, label_entry->d_name) == -1)
            exit(1);

        stat(next_src_label_path, &st);
        if (S_ISDIR(st.st_mode)) {
            debug("%*srecursing into %s\n", indent, "", next_src_label_path);
            indent += 2;
            directory_recurse(next_src_label_path, next_src_depth_path,
                              next_dest_label_path, next_dest_depth_path);
            indent -= 2;

            free(next_src_label_path);
            free(next_dest_label_path);
            free(next_src_depth_path);
            free(next_dest_depth_path);
        } else if ((ext = strstr(label_entry->d_name, ".png")) && ext[4] == '\0') {
            struct work work;

            work.src_label_path = next_src_label_path;
            work.dst_label_path = next_dest_label_path;
            work.src_depth_path = next_src_depth_path;
            work.dst_depth_path = next_dest_depth_path;

            work_queue.push(work);
        }
    }

    closedir(label_dir);
}

static void *
worker_thread_cb(void *data)
{
    debug("Running worker thread\n");

    for (;;) {
        struct work work;

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

        debug("Thread %d: processing %s\n", state->idx, work.src_label_path);
        process_png_file(work.src_label_path, work.dst_label_path);

        strcpy(work.src_depth_path + strlen(work.src_depth_path) - 4, ".exr");
        strcpy(work.dst_depth_path + strlen(work.dst_depth_path) - 4, ".exr");

        debug("Thread %d: processing %s\n", state->idx, work.src_depth_path);
        process_exr_file(work.src_depth_path, work.dst_depth_path);

        free(work.src_label_path);
        free(work.src_depth_path);
        free(work.dst_label_path);
        free(work.dst_depth_path);
    }

    return NULL;
}

static bool
read_file(const char *filename, void *buf, int max)
{
    int fd;
    int n;

    memset(buf, 0, max);

    fd = open(filename, 0);
    if (fd < 0)
        return false;
    n = read(fd, buf, max - 1);
    close(fd);
    if (n < 0)
        return false;

    return true;
}

static void
cpu_count_once_cb(void)
{
    char buf[32];
    unsigned ignore = 0, max_cpu = 0;

    if (!read_file("/sys/devices/system/cpu/present", buf, sizeof(buf)))
        return;

    if (sscanf(buf, "%u-%u", &ignore, &max_cpu) != 2)
        return;

    n_cpus = max_cpu + 1;
}

static int
cpu_count(void)
{
    pthread_once(&cpu_count_once, cpu_count_once_cb);

    return n_cpus;
}

int
main(int argc, char **argv)
{
    char *label_dest, *depth_dest;

    if (argc != 4) {
        fprintf(stderr, "Usage: program_name <top_label_dir> <top_depth_dir> <dest>\n");
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

    grey_to_id_map[0x40] = 33; // background

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

    if (asprintf(&label_dest, "%s/labels", argv[3]) == -1)
        exit(1);
    if (asprintf(&depth_dest, "%s/depth", argv[3]) == -1)
        exit(1);

    printf("Queuing frames to process...\n");

    uint64_t start = get_time();
    directory_recurse(argv[1], argv[2], label_dest, depth_dest);
    uint64_t end = get_time();

    uint64_t duration_ns = end - start;
    printf("%d frames queued to process, in %.3f%s\n",
           (int)work_queue.size(),
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    int n_cpus = cpu_count();
    //n_cpus = 1;

    printf("Spawning %d worker threads\n", n_cpus);

    workers.resize(n_cpus, worker_state());

    start = get_time();

    for (int i = 0; i < n_cpus; i++) {
        workers[i].idx = i;
        pthread_create(&workers[i].thread,
                       NULL, //sttributes
                       worker_thread_cb,
                       &workers[i]); //data
    }

    for (int i = 0; i < n_cpus; i++) {
        void *ret;

        pthread_join(workers[i].thread, &ret);
    }

    end = get_time();
    duration_ns = end - start;

    printf("Finished processing all frames in %.3f%s\n",
           get_duration_ns_print_scale(duration_ns),
           get_duration_ns_print_scale_suffix(duration_ns));

    return 0;
}
