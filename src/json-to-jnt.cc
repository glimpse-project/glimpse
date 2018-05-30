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

struct joint_mapping {
    char *name;
    const char *end; // "head" or "tail"
};

struct data {
    struct gm_logger *log;
    std::vector<joint_mapping> joint_map;
    int n_joints;
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
    std::vector<joint_mapping> &joint_map = data->joint_map;

    char json_filename[512];
    xsnprintf(json_filename, sizeof(json_filename), "%s/labels/%s.json", top_dir, path);

    JSON_Value *frame_js = json_parse_file(json_filename);
    if (!frame_js) {
        gm_throw(data->log, err, "Failed to parse %s", json_filename);
        return false;
    }

    JSON_Array *bones = json_object_get_array(json_object(frame_js), "bones");
    int n_bones = json_array_get_count(bones);

    int n_joints = data->n_joints;
    float jnt_data[n_joints * 3];

    for (int i = 0; i < n_joints; i++) {
        struct joint_mapping joint = joint_map[i];
        int jnt_pos = i * 3;

        bool found = false;
        for (int j = 0; j < n_bones; j++) {
            JSON_Object *bone = json_array_get_object(bones, j);
            const char *bone_name = json_object_get_string(bone, "name");
            if (strcmp(joint.name, bone_name) != 0)
                continue;

            JSON_Array *end = json_object_get_array(bone, joint.end);
            if (!end)
                break;

            jnt_data[jnt_pos+0] = json_array_get_number(end, 0);
            jnt_data[jnt_pos+1] = json_array_get_number(end, 1);
            jnt_data[jnt_pos+2] = json_array_get_number(end, 2);

            //printf("%s.%s = (%5.2f, %5.2f, %5.2f)\n",
            //       joint.name, joint.end,
            //       jnt_data[jnt_pos+0],
            //       jnt_data[jnt_pos+1],
            //       jnt_data[jnt_pos+2]);
            found  = true;
            break;
        }

        if (!found) {
            gm_throw(data->log, err, "Failed to find bone %s.%s in %s",
                     joint.name, joint.end, json_filename);
            json_value_free(frame_js);
            return false;
        }
    }

    char jnt_filename[512];
    xsnprintf(jnt_filename, sizeof(jnt_filename), "%s/labels/%s.jnt", top_dir, path);

    FILE *fp = fopen(jnt_filename, "w");
    if (!fp) {
        gm_throw(data->log, err, "Failed to open %s for writing", jnt_filename);
        json_value_free(frame_js);
        return false;
    }

    if (fwrite(jnt_data, sizeof(jnt_data), 1, fp) != 1) {
        gm_throw(data->log, err, "Failed to write joint data to %s", jnt_filename);
        fclose(fp);
        json_value_free(frame_js);
        return false;
    }

    fclose(fp);

    json_value_free(frame_js);
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
    JSON_Value *joint_map = json_parse_file(joint_map_file);
    if (!joint_map) {
        gm_error(data.log, "Failed to parse %s", joint_map_file);
        return 1;
    }
    int n_joints = json_array_get_count(json_array(joint_map));
    data.n_joints = n_joints;

    JSON_Array *joint_map_arr = json_array(joint_map);

    for (int i = 0; i < n_joints; i++) {
        JSON_Object *joint = json_array_get_object(joint_map_arr, i);

        const char *name = json_object_get_string(joint, "joint");
        const char *dot = strstr(name, ".");
        if (!dot) {
            gm_error(data.log, "Spurious joint %s in %s not formatted like <name>.<end>",
                     name, joint_map_file);
            return 1;
        }
        int name_len = dot - name;
        struct joint_mapping mapping;
        mapping.name = strndup(name, name_len);
        mapping.end = name + name_len + 1;
        printf("joint %d: \"%s\" %s\n", i, mapping.name, mapping.end);
        data.joint_map.push_back(mapping);
    }

    struct gm_data_index *data_index =
        gm_data_index_open(data.log,
                           argv[optind], // data dir
                           argv[optind+1], // index name
                           NULL); // abort on error

    gm_data_index_foreach(data_index,
                          write_jnt_file_foreach_cb,
                          &data, // user data
                          NULL); // abort on error

    gm_data_index_destroy(data_index);

    for (int i = 0; i < n_joints; i++) {
        free(data.joint_map[i].name);
    }
    json_value_free(joint_map);

    gm_logger_destroy(data.log);

    return 0;
}
