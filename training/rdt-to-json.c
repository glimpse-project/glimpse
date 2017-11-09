#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <getopt.h>

#include "xalloc.h"
#include "loader.h"
#include "parson.h"

static uint8_t *
xread_file(const char *filename, int *len)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(1);
    }

    struct stat sb;
    if (fstat(fileno(fp), &sb) < 0) {
        fprintf(stderr, "Failed to stat %s file descriptor: %m\n", filename);
        exit(1);
    }

    uint8_t *data = xmalloc(sb.st_size);
    if (fread(data, sb.st_size, 1, fp) != 1) {
        fprintf(stderr, "Failed to read %s\n", filename);
        exit(1);
    }

    fclose(fp);

    *len = sb.st_size;

    return data;
}

static JSON_Value *
recursive_build_tree(RDTree *tree, Node *node, int depth, int id)
{
    JSON_Value *json_node_val = json_value_init_object();
    JSON_Object *json_node = json_object(json_node_val);

    if (node->label_pr_idx == 0) {
        json_object_set_number(json_node, "t", node->t);

        JSON_Value *u_val = json_value_init_array();
        JSON_Array *u = json_array(u_val);
        json_array_append_number(u, node->uv[0]);
        json_array_append_number(u, node->uv[1]);
        json_object_set_value(json_node, "u", u_val);

        JSON_Value *v_val = json_value_init_array();
        JSON_Array *v = json_array(v_val);
        json_array_append_number(v, node->uv[2]);
        json_array_append_number(v, node->uv[3]);
        json_object_set_value(json_node, "v", v_val);

        if (depth < (tree->header.depth - 1)) {
            /* NB: The nodes in .rdt files are in a packed array arranged in
             * breadth-first, left then right child order with the root node at
             * index zero.
             *
             * With this layout then given an index for any particular node
             * ('id' here) then 2 * id + 1 is the index for the left child and
             * 2 * id + 2 is the index for the right child...
             */
            int left_id = id * 2 + 1;
            Node *left_node = tree->nodes + left_id;
            int right_id = id * 2 + 2;
            Node *right_node = tree->nodes + right_id;

            JSON_Value *left_json = recursive_build_tree(tree, left_node,
                                                         depth + 1, left_id);
            json_object_set_value(json_node, "l", left_json);
            JSON_Value *right_json = recursive_build_tree(tree, right_node,
                                                         depth + 1, right_id);
            json_object_set_value(json_node, "r", right_json);
        }
    } else {
        JSON_Value *probs_val = json_value_init_array();
        JSON_Array *probs = json_array(probs_val);

        /* NB: node->label_pr_idx is a base-one index since index zero is
         * reserved to indicate that the node is not a leaf node
         */
        float *pr_table = &tree->label_pr_tables[(node->label_pr_idx - 1) *
                                                 tree->header.n_labels];

        for (int i = 0; i < tree->header.n_labels; i++)
            json_array_append_number(probs, pr_table[i]);

        json_object_set_value(json_node, "p", probs_val);
    }

    return json_node_val;
}

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

    int tree_data_len = 0;
    uint8_t *tree_data = xread_file(argv[optind], &tree_data_len);

    RDTree *tree = load_tree(tree_data, tree_data_len);

    free(tree_data);

    JSON_Value *root = json_value_init_object();

    /* just in case it's handy for working around anything later */
    json_object_set_number(json_object(root), "_rdt_version_was", tree->header.version);

    json_object_set_number(json_object(root), "depth", tree->header.depth);
    json_object_set_number(json_object(root), "vertical_fov", tree->header.fov);
    json_object_set_number(json_object(root), "n_labels", tree->header.n_labels);
    json_object_set_number(json_object(root), "bg_label", tree->header.bg_label);

    JSON_Value *nodes = recursive_build_tree(tree, tree->nodes, 0, 0);

    json_object_set_value(json_object(root), "root", nodes);

    JSON_Status status;

    if (pretty)
        status = json_serialize_to_file_pretty(root, argv[optind+1]);
    else
        status = json_serialize_to_file(root, argv[optind+1]);

    if (status != JSONSuccess) {
        fprintf(stderr, "Failed to serialize output to JSON\n");
        return 1;
    }

    return 0;
}
