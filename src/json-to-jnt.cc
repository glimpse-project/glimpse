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

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <getopt.h>

#include <vector>

#include <glimpse_log.h>

#include "train_utils.h"

#define xsnprintf(dest, size, fmt, ...) do { \
        if (snprintf(dest, size, fmt,  __VA_ARGS__) >= (int)size) \
            exit(1); \
    } while(0)

struct data {
    struct gm_logger *log;
    int n_joints;
    float *joint_data;
};

static bool
write_jnt_file_foreach_cb(struct gm_data_index* data_index,
                          int index,
                          const char* path,
                          void* user_data,
                          char** err)
{
    struct data *data = (struct data *)user_data;
    const char* top_dir = gm_data_index_get_top_dir(data_index);

    char jnt_filename[512];
    xsnprintf(jnt_filename, sizeof(jnt_filename), "%s/labels/%s.jnt", top_dir, path);

    FILE *fp = fopen(jnt_filename, "w");
    if (!fp) {
        gm_throw(data->log, err, "Failed to open %s for writing", jnt_filename);
        return false;
    }

    int n_joints = data->n_joints;
    int64_t off = (int64_t)index * n_joints * 3;
    float *jnt_data = &data->joint_data[off];
    int jnt_data_size = n_joints * 3 * sizeof(float);

    printf("writting %d bytes to %s\n", jnt_data_size, jnt_filename);
    if (fwrite(jnt_data, jnt_data_size, 1, fp) != 1) {
        gm_throw(data->log, err, "Failed to write joint data to %s", jnt_filename);
        fclose(fp);
        return false;
    }

    fclose(fp);

    return true;
}

static void
usage(void)
{
    fprintf(stderr,
"Usage json-to-rdt [options] <data dir> <index name> <joint map>\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n"
"Writes .jnt files for every frame in the loaded training data index\n"
    );
}

int
main(int argc, char **argv)
{
    struct data data;
    data.log = gm_logger_new(NULL, NULL);

    const char *short_options="h";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return 1;
        }
    }

    if (argc - optind != 3) {
        usage();
        return 1;
    }

    const char *joint_map_file = argv[optind+2];

    struct gm_data_index *data_index =
        gm_data_index_open(data.log,
                           argv[optind], // data dir
                           argv[optind+1], // index name
                           NULL); // abort on error

    gm_data_index_load_joints(data_index,
                              joint_map_file,
                              &data.n_joints,
                              &data.joint_data,
                              NULL); // abort on error
    gm_data_index_foreach(data_index,
                          write_jnt_file_foreach_cb,
                          &data, // user data
                          NULL); // abort on error

    xfree(data.joint_data);

    gm_data_index_destroy(data_index);

    gm_logger_destroy(data.log);

    return 0;
}
