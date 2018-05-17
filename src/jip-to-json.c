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

#include "xalloc.h"
#include "jip.h"
#include "parson.h"


static void
usage(void)
{
    printf(
"Usage rdt-to-json [options] <in.rdt> <out.json>\n"
"\n"
"    -p,--pretty                Pretty print the JSON output\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n"
"This tool converts the binary representation of .jip files output by the\n"
"train_joint_params tool to a JSON representation which can be convenient\n"
"for loading the trees in Python or JavaScript tools or inspecting manually.\n"
"\n"
"The schema looks like: FIXME\n"
"\n"
    );

    exit(1);
}

int
main(int argc, char **argv)
{
    int opt;
    bool pretty = false;

    const char *short_options="+hp";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {"pretty",          no_argument,        0, 'p'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
            case 'h':
                usage();
                return 0;
            case 'p':
                pretty = true;
                break;
            default:
                usage();
                break;
        }
    }

    if (argc - optind != 2)
        usage();

    JIParams *params = jip_load_from_file(argv[optind]);

    JSON_Value *jip = json_value_init_object();

    int n_joints = params->header.n_joints;
    json_object_set_number(json_object(jip), "n_joints", n_joints);

    JSON_Value *jip_params = json_value_init_array();
    for (int i = 0; i < n_joints; i++) {
        JIParam *param = params->joint_params + i;
        JSON_Value *jip_param = json_value_init_object();

        json_object_set_number(json_object(jip_param), "bandwidth", param->bandwidth);
        json_object_set_number(json_object(jip_param), "threshold", param->threshold);
        json_object_set_number(json_object(jip_param), "offset", param->offset);

        json_array_append_value(json_array(jip_params), jip_param);
    }

    json_object_set_value(json_object(jip), "params", jip_params);

    if (pretty)
        json_serialize_to_file_pretty(jip, argv[optind+1]);
    else
        json_serialize_to_file(jip, argv[optind+1]);

    jip_free(params);

    return 0;
}
