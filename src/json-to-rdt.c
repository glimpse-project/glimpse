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
"Usage rdt-to-json [options] <in.json> <out.rdt>\n"
"\n"
"    -h,--help                  Display this help\n\n"
"\n"
"This tool converts the JSON representation of the randomised decision trees\n"
"output by the train_rdt tool to a binary representation which can be\n"
"convenient for fast loading of trees and more compact representation when\n"
"compressed.\n"
    );
}

int
main(int argc, char **argv)
{
    int opt;
    const char *short_options="+hp";
    const struct option long_options[] = {
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

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

    if (argc - optind != 2) {
        usage();
        return 1;
    }

    RDTree *tree = read_json_tree(argv[optind]);
    if (!tree) return 1;
    return save_tree(tree, argv[optind+1]) ?
      0 : 1;
}
