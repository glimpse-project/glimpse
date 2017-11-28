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

#include "loader.h"

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
"This tool converts the binary representation of the randomized decision trees\n"
"output by the train_rdt tool to a JSON representation which can be convenient\n"
"for loading the trees in Python or JavaScript tools or inspecting manually.\n"
"\n"
"The schema looks like:\n"
"\n"
"  {\n"
"    \"n_labels\": 34,\n"
"    \"vertical_fov\": 50.4,\n"
"    \"depth\": 20,\n"
"    \"root\": {\n"
"      \"t\": 0.5,            // threshold\n"
"      \"u\": [  3, -0.5 ],   // U vector\n"
"      \"v\": [ -2, 10.1 ],   // V vector\n"
"\n"
"      // left node...\n"
"      \"l\": {\n"
"        // children directly nested...\n"
"        \"t\": 0.2,          // child's threshold\n"
"        \"u\": [  1, -5.2 ], // child's U vector\n"
"        \"v\": [ -7, -0.1 ], // child's V vector\n"
"        \"l\": { ... },      // child's left node\n"
"        \"r\": { ... },      // child's right node\n"
"      },\n"
"\n"
"      // right node is an example leaf node with <n_labels> label\n"
"      // probabilities...\n"
"      \"r\": {\n"
"        \"p\": [0, 0, 0, 0.2, 0, 0.2, 0, 0.6, 0, 0 ... ],\n"
"      }\n"
"    }\n"
"  }\n"
    );
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
                return 1;
        }
    }

    if (argc - optind != 2) {
        usage();
        return 1;
    }

    RDTree *tree = read_tree(argv[optind]);
    if (!tree) return 1;
    return save_tree_json(tree, argv[optind+1], pretty) ?
      0 : 1;
}
