/*
 * Copyright (C) 2018 Glimp IP Ltd
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
#include <unistd.h>
#include <stdio.h>
#include <getopt.h>

#include <vector>

#include "png.h"

#include "rdt_tree.h"

#include "glimpse_data.h"

static bool verbose_opt = false;

/* XXX: Copied from image-pre-processor */
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

static uint64_t
get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void
usage(void)
{
    fprintf(stderr,
"Usage: index-to-recording [OPTIONS] <data dir> <index name> <out dir>\n"
"\n"
"Creates a glimpse_viewer recording based on a given index of pre-processed\n"
"training data frames\n"
"\n"
"  -v, --verbose                 Verbose output.\n"
"  -h, --help                    Display this message.\n"
    );
    exit(1);
}

int
main(int argc, char **argv)
{
    struct gm_logger *log = gm_logger_new(NULL, NULL);

    const char *short_options="vht";
    const struct option long_options[] = {
        {"verbose",         no_argument,        0, 'v'},
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
        case 'v':
            verbose_opt = true;
            break;
        case 'h':
            usage();
            break;
        default:
            usage();
            break;
        }
    }

    if (argc - optind < 3)
        usage();

    const char *data_dir = argv[optind];
    const char *index_name = argv[optind + 1];
    const char *out_dir = argv[optind + 2];

    char out_filename[512];
    snprintf(out_filename, sizeof(out_filename), "%s/glimpse_recording.json", out_dir);
    struct stat sb;
    if (stat(out_filename, &sb) == 0) {
        gm_error(log, "%s already exists", out_filename);
        exit(1);
    }

    int ret = mkdir(out_dir, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Failed to ensure top-level directory exists for recordings");
        exit(1);
    }

    char depth_dir[512];
    snprintf(depth_dir, sizeof(depth_dir), "%s/depth", out_dir);
    ret = mkdir(depth_dir, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Failed to create %s directory", depth_dir);
        exit(1);
    }

    char video_dir[512];
    snprintf(video_dir, sizeof(video_dir), "%s/video", out_dir);
    ret = mkdir(video_dir, 0777);
    if (ret < 0 && errno != EEXIST) {
        gm_error(log, "Failed to create %s directory", video_dir);
        exit(1);
    }

    int width;
    int height;
    int n_images;

    float *depth_images;
    uint8_t *label_images;

    JSON_Value *meta =
        gm_data_load_simple(log,
                            data_dir,
                            index_name,
                            NULL, // no joint map
                            &n_images,
                            NULL, // n_joints
                            &width,
                            &height,
                            &depth_images,
                            &label_images,
                            NULL, // no joint data
                            NULL); // abort on error
    JSON_Object* camera = json_object_get_object(json_object(meta), "camera");
    float vfov = json_object_get_number(camera, "vertical_fov") * (M_PI / 180.0);

    JSON_Value *recording = json_value_init_object();

    JSON_Value *intrinsics = json_value_init_object();
    json_object_set_number(json_object(intrinsics), "width", width);
    json_object_set_number(json_object(intrinsics), "height", height);

    float fy = 0.5 * height / tanf(vfov / 2.0);
    float fx = fy;
    json_object_set_number(json_object(intrinsics), "fx", fx);
    json_object_set_number(json_object(intrinsics), "fy", fy);

    json_object_set_number(json_object(intrinsics), "cx", (float)width / 2);
    json_object_set_number(json_object(intrinsics), "cy", (float)height / 2);
    json_object_set_number(json_object(intrinsics), "distortion_model", 0);

    json_object_set_value(json_object(recording), "depth_intrinsics", intrinsics);
    json_object_set_value(json_object(recording), "video_intrinsics",
                          json_value_deep_copy(intrinsics));

    JSON_Value *extrinsics = json_value_init_object();
    JSON_Value *rot33 = json_value_init_array();
    for (int i = 0; i < 9; i++)
        json_array_append_number(json_array(rot33), 0);
    json_object_set_value(json_object(extrinsics), "rotation", rot33);
    JSON_Value *translate = json_value_init_array();
    for (int i = 0; i < 3; i++)
        json_array_append_number(json_array(translate), 0);
    json_object_set_value(json_object(extrinsics), "translation", translate);

    json_object_set_value(json_object(recording), "depth_to_video_extrinsics", extrinsics);

    json_object_set_number(json_object(recording), "depth_format", 2); // f32
    json_object_set_number(json_object(recording), "video_format", 7); // rgba u8
    JSON_Value *frames = json_value_init_array();
    json_object_set_value(json_object(recording), "frames", frames);

    uint64_t current_time = get_time();

    uint8_t *rgba_image = (uint8_t *)xmalloc(width * height * 4);
    for (int i = 0; i < n_images; i++) {
        int64_t img_off = (int64_t)i * width * height;
        float *depth = &depth_images[img_off];
        uint8_t *labels = &label_images[img_off];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pos = y * width + x;
                uint8_t label = labels[pos];

                rgba_image[pos*4+0] = palette[label].red;
                rgba_image[pos*4+1] = palette[label].green;
                rgba_image[pos*4+2] = palette[label].blue;
                rgba_image[pos*4+3] = 0xff;
            }
        }

        JSON_Value *frame = json_value_init_object();
        json_object_set_number(json_object(frame), "timestamp", current_time);
        current_time += 16000000;

        char bin_filename[512];
        snprintf(bin_filename, sizeof(bin_filename), "%s/depth/frame%06d.bin",
                 out_dir, i);
        FILE *fp = fopen(bin_filename, "w");
        gm_assert(log, fp != NULL, "Failed to open %s", bin_filename);
        if (fwrite(depth, width * height * 4, 1, fp) != 1) {
            gm_error(log, "Failed to write %s", bin_filename);
            exit(1);
        }
        fclose(fp);
        fp = NULL;
        snprintf(bin_filename, sizeof(bin_filename), "/depth/frame%06d.bin", i);
        json_object_set_string(json_object(frame), "depth_file", bin_filename);
        json_object_set_number(json_object(frame), "depth_len", width * height * 4);

        snprintf(bin_filename, sizeof(bin_filename), "%s/video/frame%06d.bin",
                 out_dir, i);
        fp = fopen(bin_filename, "w");
        gm_assert(log, fp != NULL, "Failed to open %s", bin_filename);
        if (fwrite(rgba_image, width * height * 4, 1, fp) != 1) {
            gm_error(log, "Failed to write %s", bin_filename);
            exit(1);
        }
        fclose(fp);
        snprintf(bin_filename, sizeof(bin_filename), "/video/frame%06d.bin", i);
        json_object_set_string(json_object(frame), "video_file", bin_filename);
        json_object_set_number(json_object(frame), "video_len", width * height * 4);

        json_object_set_number(json_object(frame), "camera_rotation", 0);

        json_array_append_value(json_array(frames), frame);
    }

    xfree(rgba_image);

    json_serialize_to_file_pretty(recording, out_filename);
    json_value_free(recording);

    return 0;
}
