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

#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cstddef>

#include "rdt_tree.h"
#include "parson.h"
#include "xalloc.h"

static void
assert_rdt_abi()
{
    static_assert(sizeof(RDTHeader) == 11, "RDT ABI Breakage");
    static_assert(sizeof(Node) == 32,      "RDT ABI Breakage");
    static_assert(offsetof(Node, t) == 16, "RDT ABI Breakage");
}

static int
count_pr_tables(JSON_Object* node)
{
    if (!node)
        return 0;

    if (json_object_has_value(node, "p"))
        return 1;

    return count_pr_tables(json_object_get_object(node, "l")) +
        count_pr_tables(json_object_get_object(node, "r"));
}

static void
unpack_json_tree(JSON_Object* jnode, Node* nodes, int node_index,
                 float* pr_tables, int* table_index, int n_labels)
{
    Node* node = &nodes[node_index];

    if (json_object_has_value(jnode, "p"))
    {
        float* pr_table = &pr_tables[(*table_index) * n_labels];

        JSON_Array* p = json_object_get_array(jnode, "p");
        for (int i = 0; i < n_labels; i++)
        {
            pr_table[i] = (float)json_array_get_number(p, i);
        }

        // Write out probability table
        node->label_pr_idx = ++(*table_index);
        return;
    }

    JSON_Array* u = json_object_get_array(jnode, "u");
    JSON_Array* v = json_object_get_array(jnode, "v");

    if (u == NULL || v == NULL) {
        node->label_pr_idx = INT_MAX;
        return;
    }

    node->uv[0] = json_array_get_number(u, 0);
    node->uv[1] = json_array_get_number(u, 1);
    node->uv[2] = json_array_get_number(v, 0);
    node->uv[3] = json_array_get_number(v, 1);
    node->t = json_object_get_number(jnode, "t");
    node->label_pr_idx = 0;

    unpack_json_tree(json_object_get_object(jnode, "l"), nodes,
                     node_index * 2 + 1, pr_tables, table_index, n_labels);
    unpack_json_tree(json_object_get_object(jnode, "r"), nodes,
                     node_index * 2 + 2, pr_tables, table_index, n_labels);
}

void
rdt_tree_destroy(RDTree* tree)
{
    if (tree->nodes)
    {
        xfree(tree->nodes);
    }
    if (tree->label_pr_tables)
    {
        xfree(tree->label_pr_tables);
    }
    xfree(tree);
}

RDTree*
rdt_tree_load_from_json(struct gm_logger* log,
                        JSON_Value* json_tree_value,
                        char** err)
{
    assert_rdt_abi();

    JSON_Object* json_tree = json_value_get_object(json_tree_value);
    if (!json_tree)
    {
        gm_throw(log, err, "Failed to find top-level tree object\n");
        json_value_free(json_tree_value);
        return NULL;
    }

    JSON_Object* root = json_object_get_object(json_tree, "root");
    if (!root)
    {
        gm_throw(log, err, "Failed to find tree root node\n");
        json_value_free(json_tree_value);
        return NULL;
    }

    RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));
    tree->header.tag[0] = 'R';
    tree->header.tag[1] = 'D';
    tree->header.tag[2] = 'T';
    tree->header.version = RDT_VERSION;

    tree->header.depth = (uint8_t)json_object_get_number(json_tree, "depth");
    tree->header.n_labels = (uint8_t)json_object_get_number(json_tree, "n_labels");
    tree->header.bg_label = (uint8_t)json_object_get_number(json_tree, "bg_label");
    tree->header.fov = (float)json_object_get_number(json_tree, "vertical_fov");

    // Count probability arrays
    int n_pr_tables = root ? count_pr_tables(root) : 0;
    tree->n_pr_tables = n_pr_tables;

    // Allocate tree structure
    int n_nodes = (1<<tree->header.depth) - 1;
    tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));

    /* In case we don't have a complete tree we need to initialize label_pr_idx
     * to imply that the node has not been trained yet
     */
    for (int i = 0; i < n_nodes; i++)
        tree->nodes[i].label_pr_idx = INT_MAX;

    tree->label_pr_tables = (float*)
        xmalloc(n_pr_tables * tree->header.n_labels * sizeof(float));

    // Copy over nodes and probability tables
    int table_index = 0;
    unpack_json_tree(root, tree->nodes, 0, tree->label_pr_tables, &table_index,
                     tree->header.n_labels);

    return tree;
}

RDTree*
rdt_tree_load_from_json_file(struct gm_logger* log,
                             const char* filename,
                             char** err)
{
    JSON_Value *js = json_parse_file(filename);
    if (!js) {
        gm_throw(log, err, "Failed to parse %s", filename);
        return NULL;
    }

    RDTree *tree = rdt_tree_load_from_json(log, js, err);
    json_value_free(js);

    return tree;
}

RDTree*
rdt_tree_load_from_buf(struct gm_logger* log,
                       uint8_t* tree_buf,
                       int len,
                       char** err)
{
    assert_rdt_abi();

    RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

    if ((size_t)len < sizeof(RDTHeader))
    {
        fprintf(stderr, "Buffer too small to contain tree\n");
        rdt_tree_destroy(tree);
        return NULL;
    }
    memcpy(&tree->header, tree_buf, sizeof(RDTHeader));
    tree_buf += sizeof(RDTHeader);
    len -= sizeof(RDTHeader);

    if (strncmp(tree->header.tag, "RDT", 3) != 0)
    {
        fprintf(stderr, "File is not an RDT file\n");
        rdt_tree_destroy(tree);
        return NULL;
    }

    if (tree->header.version != RDT_VERSION)
    {
        fprintf(stderr, "Incompatible RDT version, expected %u, found %u\n",
                RDT_VERSION, (unsigned)tree->header.version);
        rdt_tree_destroy(tree);
        return NULL;
    }

    // Read in the decision tree nodes
    int n_nodes = (1<<tree->header.depth) - 1;
    tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
    if ((size_t)len < (sizeof(Node) * n_nodes))
    {
        fprintf(stderr, "Error parsing tree nodes\n");
        rdt_tree_destroy(tree);
        return NULL;
    }
    memcpy(tree->nodes, tree_buf, sizeof(Node) * n_nodes);
    tree_buf += sizeof(Node) * n_nodes;
    len -= sizeof(Node) * n_nodes;

    // Read in the label probabilities
    int label_bytes = len;
    if (label_bytes % sizeof(float) != 0)
    {
        fprintf(stderr, "Unexpected size of label probability tables\n");
        rdt_tree_destroy(tree);
        return NULL;
    }
    int n_prs = label_bytes / sizeof(float);
    if (n_prs % tree->header.n_labels != 0)
    {
        fprintf(stderr, "Unexpected number of label probabilities\n");
        rdt_tree_destroy(tree);
        return NULL;
    }
    int n_tables = n_prs / tree->header.n_labels;

    tree->n_pr_tables = n_tables;
    tree->label_pr_tables = (float*)xmalloc(label_bytes);
    memcpy(tree->label_pr_tables, tree_buf,
           sizeof(float) * tree->header.n_labels * n_tables);

    return tree;
}

RDTree*
rdt_tree_load_from_file(struct gm_logger* log,
                        const char* filename,
                        char** err)
{
    FILE* tree_fp;
    if (!(tree_fp = fopen(filename, "r")))
    {
        gm_throw(log, err, "Failed to open decision tree: %s\n", filename);
        return NULL;
    }

    struct stat sb;
    if (fstat(fileno(tree_fp), &sb) < 0)
    {
        gm_throw(log, err, "Failed to stat decision tree: %s\n", filename);
        fclose(tree_fp);
        return NULL;
    }

    uint8_t* tree_buf = (uint8_t*)xcalloc(1, sb.st_size);
    if (fread(tree_buf, sb.st_size, 1, tree_fp) != 1)
    {
        gm_throw(log, err, "Failed to read decision tree: %s\n", filename);
        xfree(tree_buf);
        fclose(tree_fp);
        return NULL;
    }

    RDTree* tree = rdt_tree_load_from_buf(log, tree_buf, sb.st_size, err);
    xfree(tree_buf);
    fclose(tree_fp);

    return tree;
}

bool
rdt_tree_save(RDTree* tree, const char* filename)
{
    int n_nodes;
    bool success;
    FILE* output;

    if (!(output = fopen(filename, "wb")))
    {
        fprintf(stderr, "Failed to open output file '%s'\n", filename);
        return false;
    }

    success = false;
    if (fwrite(&tree->header, sizeof(RDTHeader), 1, output) != 1)
    {
        fprintf(stderr, "Error writing header\n");
        goto save_tree_close;
    }

    n_nodes = (1<<tree->header.depth) - 1;
    if (fwrite(tree->nodes, sizeof(Node), n_nodes, output) != (size_t)n_nodes)
    {
        fprintf(stderr, "Error writing tree nodes\n");
        goto save_tree_close;
    }

    if (fwrite(tree->label_pr_tables, sizeof(float) * tree->header.n_labels,
               tree->n_pr_tables, output) != (size_t)tree->n_pr_tables)
    {
        fprintf(stderr, "Error writing tree probability tables\n");
        goto save_tree_close;
    }

    success = true;

save_tree_close:
    if (fclose(output) != 0)
    {
        fprintf(stderr, "Error closing output file\n");
        return false;
    }

    return success;
}

static bool
check_forest_consistency(struct gm_logger* log,
                         RDTree** forest,
                         int n_trees,
                         char** err)
{
    int n_labels = forest[0]->header.n_labels;

    for (int i = 0; i < n_trees; i++) {
        RDTree* tree = forest[i];

        if (tree->header.n_labels != n_labels) {
            gm_throw(log, err, "Tree %d has %d labels, expected %d\n",
                     i, tree->header.n_labels,
                     n_labels);
            return false;
        }
    }

    return true;
}

RDTree**
rdt_forest_load_from_files(struct gm_logger* log,
                           const char** files,
                           int n_files,
                           char** err)
{
    RDTree** trees = (RDTree**)xcalloc(n_files, sizeof(RDTree*));

    for (int i = 0; i < n_files; i++) {
        trees[i] = rdt_tree_load_from_file(log, files[i], err);
        if (!trees[i]) {
            rdt_forest_destroy(trees, n_files);
            return NULL;
        }
    }

    if (!check_forest_consistency(log, trees, n_files, err)) {
        rdt_forest_destroy(trees, n_files);
        return NULL;
    }

    return trees;
}

void
rdt_forest_destroy(RDTree** forest, int n_trees)
{
    for (int i = 0; i < n_trees; i++) {
        if (forest[i])
            rdt_tree_destroy(forest[i]);
    }
    xfree(forest);
}
