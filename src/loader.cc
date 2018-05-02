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
#include <string.h>
#include <math.h>
#include <cstddef>

#include "loader.h"
#include "parson.h"
#include "xalloc.h"

static void
assert_rdt_abi()
{
  static_assert(sizeof(RDTHeader) == 11, "RDT ABI Breakage");
  static_assert(sizeof(Node) == 32,      "RDT ABI Breakage");
  static_assert(offsetof(Node, t) == 16, "RDT ABI Breakage");
}

bool
save_tree(RDTree* tree, const char* filename)
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

  n_nodes = roundf(powf(2.f, tree->header.depth)) - 1;
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

static JSON_Value*
recursive_build_tree(RDTree* tree, Node* node, int depth, int id)
{
  JSON_Value* json_node_val = json_value_init_object();
  JSON_Object* json_node = json_object(json_node_val);

  if (node->label_pr_idx == 0)
    {
      json_object_set_number(json_node, "t", node->t);

      JSON_Value* u_val = json_value_init_array();
      JSON_Array* u = json_array(u_val);
      json_array_append_number(u, node->uv[0]);
      json_array_append_number(u, node->uv[1]);
      json_object_set_value(json_node, "u", u_val);

      JSON_Value* v_val = json_value_init_array();
      JSON_Array* v = json_array(v_val);
      json_array_append_number(v, node->uv[2]);
      json_array_append_number(v, node->uv[3]);
      json_object_set_value(json_node, "v", v_val);

      if (depth < (tree->header.depth - 1))
        {
          /* NB: The nodes in .rdt files are in a packed array arranged in
           * breadth-first, left then right child order with the root node at
           * index zero.
           *
           * With this layout then given an index for any particular node
           * ('id' here) then 2 * id + 1 is the index for the left child and
           * 2 * id + 2 is the index for the right child...
           */
          int left_id = id * 2 + 1;
          Node* left_node = tree->nodes + left_id;
          int right_id = id * 2 + 2;
          Node* right_node = tree->nodes + right_id;

          JSON_Value* left_json = recursive_build_tree(tree, left_node,
                                                       depth + 1, left_id);
          json_object_set_value(json_node, "l", left_json);
          JSON_Value* right_json = recursive_build_tree(tree, right_node,
                                                        depth + 1, right_id);
          json_object_set_value(json_node, "r", right_json);
        }
    }
  else
    {
      JSON_Value* probs_val = json_value_init_array();
      JSON_Array* probs = json_array(probs_val);

      /* NB: node->label_pr_idx is a base-one index since index zero is
       * reserved to indicate that the node is not a leaf node
       */
      float* pr_table = &tree->label_pr_tables[(node->label_pr_idx - 1) *
                                               tree->header.n_labels];

      for (int i = 0; i < tree->header.n_labels; i++)
        {
          json_array_append_number(probs, pr_table[i]);
        }

      json_object_set_value(json_node, "p", probs_val);
    }

  return json_node_val;
}

bool
save_tree_json(RDTree* tree, const char* filename, bool pretty)
{
  JSON_Value *root = json_value_init_object();

  json_object_set_number(json_object(root), "_rdt_version_was", tree->header.version);
  json_object_set_number(json_object(root), "depth", tree->header.depth);
  json_object_set_number(json_object(root), "vertical_fov", tree->header.fov);
  json_object_set_number(json_object(root), "n_labels", tree->header.n_labels);
  json_object_set_number(json_object(root), "bg_label", tree->header.bg_label);

  JSON_Value *nodes = recursive_build_tree(tree, tree->nodes, 0, 0);

  json_object_set_value(json_object(root), "root", nodes);

  JSON_Status status = pretty ?
    json_serialize_to_file_pretty(root, filename) :
    json_serialize_to_file(root, filename);

  if (status != JSONSuccess)
    {
      fprintf(stderr, "Failed to serialize output to JSON\n");
      return false;
    }

  return true;
}

static int
count_pr_tables(JSON_Object* node)
{
  if (json_object_has_value(node, "p"))
    {
      return 1;
    }

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

RDTree*
load_json_tree(uint8_t* json_tree_buf, int len)
{
  assert_rdt_abi();

  JSON_Value* json_tree_value = json_parse_string((const char *)json_tree_buf);
  if (!json_tree_value)
    {
      fprintf(stderr, "Failed to parse JSON string\n");
      return NULL;
    }

  JSON_Object* json_tree = json_value_get_object(json_tree_value);
  if (!json_tree)
    {
      fprintf(stderr, "Failed to find top-level tree object\n");
      json_value_free(json_tree_value);
      return NULL;
    }

  if ((int)json_object_get_number(json_tree, "_rdt_version_was") != RDT_VERSION)
    {
      fprintf(stderr, "Unexpected RDT version (expected %d)\n", RDT_VERSION);
      json_value_free(json_tree_value);
      return NULL;
    }

  JSON_Object* root = json_object_get_object(json_tree, "root");
  if (!root)
    {
      fprintf(stderr, "Failed to find tree root node\n");
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
  int n_pr_tables = count_pr_tables(root);
  tree->n_pr_tables = n_pr_tables;

  // Allocate tree structure
  int n_nodes = roundf(powf(2.f, tree->header.depth)) - 1;
  tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
  tree->label_pr_tables = (float*)
    xmalloc(n_pr_tables * tree->header.n_labels * sizeof(float));

  // Copy over nodes and probability tables
  int table_index = 0;
  unpack_json_tree(root, tree->nodes, 0, tree->label_pr_tables, &table_index,
                   tree->header.n_labels);

  // Free data and return tree
  json_value_free(json_tree_value);

  return tree;
}

RDTree*
read_json_tree(const char* filename)
{
  RDTree** forest = read_json_forest(&filename, 1);

  if (forest)
    {
      RDTree* tree = forest[0];
      xfree(forest);
      return tree;
    }

  return NULL;
}

RDTree*
load_tree(uint8_t* tree_buf, int len)
{
  assert_rdt_abi();

  RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

  if ((size_t)len < sizeof(RDTHeader))
    {
      fprintf(stderr, "Buffer too small to contain tree\n");
      free_tree(tree);
      return NULL;
    }
  memcpy(&tree->header, tree_buf, sizeof(RDTHeader));
  tree_buf += sizeof(RDTHeader);
  len -= sizeof(RDTHeader);

  if (strncmp(tree->header.tag, "RDT", 3) != 0)
    {
      fprintf(stderr, "File is not an RDT file\n");
      free_tree(tree);
      return NULL;
    }

  if (tree->header.version != RDT_VERSION)
    {
      fprintf(stderr, "Incompatible RDT version, expected %u, found %u\n",
              RDT_VERSION, (unsigned)tree->header.version);
      free_tree(tree);
      return NULL;
    }

  // Read in the decision tree nodes
  int n_nodes = roundf(powf(2.f, tree->header.depth)) - 1;
  tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
  if ((size_t)len < (sizeof(Node) * n_nodes))
    {
      fprintf(stderr, "Error parsing tree nodes\n");
      free_tree(tree);
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
      free_tree(tree);
      return NULL;
    }
  int n_prs = label_bytes / sizeof(float);
  if (n_prs % tree->header.n_labels != 0)
    {
      fprintf(stderr, "Unexpected number of label probabilities\n");
      free_tree(tree);
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
read_tree(const char* filename)
{
  RDTree** forest = read_forest(&filename, 1);

  if (forest)
    {
      RDTree* tree = forest[0];
      xfree(forest);
      return tree;
    }

  return NULL;
}

void
free_tree(RDTree* tree)
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

static RDTree**
load_any_forest(uint8_t** tree_bufs, int* tree_buf_lengths,
                int n_trees, bool is_json)
{
  bool error = false;
  int n_labels = 0;
  RDTree** trees = (RDTree**)xcalloc(n_trees, sizeof(RDTree*));

  for (int i = 0; i < n_trees; i++)
    {
      // Validate the decision tree
      RDTree* tree = trees[i] = is_json ?
        load_json_tree(tree_bufs[i], tree_buf_lengths[i]) :
        load_tree(tree_bufs[i], tree_buf_lengths[i]);

      if (!tree)
        {
          error = true;
          break;
        }

      if (n_labels == 0)
        {
          n_labels = tree->header.n_labels;
        }
      if (tree->header.n_labels != n_labels)
        {
          fprintf(stderr, "Tree %d has %d labels, expected %d\n",
                  i, tree->header.n_labels,
                  n_labels);
          error = true;
          break;
        }
    }

  if (error)
    {
      for (int i = 0; i < n_trees; i++)
        {
          if (trees[i])
            {
              free_tree(trees[i]);
            }
        }
      free(trees);

      return NULL;
    }

  return trees;
}

RDTree**
load_json_forest(uint8_t** tree_bufs, int* tree_buf_lengths, int n_trees)
{
  return load_any_forest(tree_bufs, tree_buf_lengths, n_trees, true);
}

RDTree**
load_forest(uint8_t** tree_bufs, int* tree_buf_lengths, int n_trees)
{
  return load_any_forest(tree_bufs, tree_buf_lengths, n_trees, false);
}

static RDTree**
read_any_forest(const char** files, int n_files, bool is_json)
{
  uint8_t* tree_bufs[n_files];
  int tree_buf_lengths[n_files];
  int n_trees = 0;
  bool error = false;

  for (int i = 0; i < n_files; i++)
    {
      const char* tree_file = files[i];

      FILE* tree_fp;
      if (!(tree_fp = fopen(tree_file, "r")))
        {
          fprintf(stderr, "Failed to open decision tree: %s\n", tree_file);
          error = true;
          break;
        }

      struct stat sb;
      if (fstat(fileno(tree_fp), &sb) < 0)
        {
          fprintf(stderr, "Failed to stat decision tree: %s\n", tree_file);
          error = true;
          break;
        }

      tree_bufs[i] = (uint8_t*)xcalloc(1, sb.st_size);
      n_trees++;
      if (fread(tree_bufs[i], sb.st_size, 1, tree_fp) != 1)
        {
          fprintf(stderr, "Failed to read decision tree: %s\n", tree_file);
          error = true;
          break;
        }
      tree_buf_lengths[i] = sb.st_size;

      if (fclose(tree_fp) != 0)
        {
          fprintf(stderr, "Error closing tree file: %s\n", tree_file);
        }
    }

  RDTree** forest = error ?
    NULL : load_any_forest(tree_bufs, tree_buf_lengths, n_trees, is_json);

  for (int i = 0; i < n_trees; i++)
    {
      xfree(tree_bufs[i]);
    }

  return forest;
}

RDTree**
read_json_forest(const char** files, int n_files)
{
  return read_any_forest(files, n_files, true);
}

RDTree**
read_forest(const char** files, int n_files)
{
  return read_any_forest(files, n_files, false);
}

void
free_forest(RDTree** forest, int n_trees)
{
  for (int i = 0; i < n_trees; i++)
    {
      free_tree(forest[i]);
    }
  xfree(forest);
}

JIParams *
joint_params_from_json(JSON_Value *root)
{
  JIParams* jip = (JIParams*)xcalloc(1, sizeof(JIParams));

  jip->header.tag[0] = 'J';
  jip->header.tag[1] = 'I';
  jip->header.tag[2] = 'P';

  jip->header.version = 0;

  jip->header.n_joints = json_object_get_number(json_object(root), "n_joints");

  JSON_Array *params = json_object_get_array(json_object(root), "params");
  int len = json_array_get_count(params);
  if (len != jip->header.n_joints)
    {
      fprintf(stderr, "Inconsistency between \"n_joints\" and length of \"params\" array\n");
      free(jip);
      return NULL;
    }

  jip->joint_params = (JIParam*)xmalloc(jip->header.n_joints * sizeof(JIParam));
  for (int i = 0; i < len; i++)
    {
      JSON_Object *param = json_array_get_object(params, i);

      jip->joint_params[i].bandwidth = json_object_get_number(param, "bandwidth");
      jip->joint_params[i].threshold = json_object_get_number(param, "threshold");
      jip->joint_params[i].offset = json_object_get_number(param, "offset");
    }

  return jip;
}

JIParams*
read_jip(const char* filename)
{
  const char* ext;

  if ((ext = strstr(filename, ".json")) && ext[5] == '\0')
    {
      JSON_Value *js = json_parse_file(filename);
      JIParams *ret = joint_params_from_json(js);
      json_value_free(js);
      return ret;
    }

  FILE* jip_file = fopen(filename, "r");
  if (!jip_file)
    {
      fprintf(stderr, "Error opening JIP file\n");
      return NULL;
    }

  JIParams* jip = (JIParams*)xcalloc(1, sizeof(JIParams));
  if (fread(&jip->header, sizeof(JIPHeader), 1, jip_file) != 1)
    {
      fprintf(stderr, "Error reading header\n");
      goto read_jip_error;
    }

  jip->joint_params = (JIParam*)xmalloc(jip->header.n_joints * sizeof(JIParam));
  for (int i = 0; i < jip->header.n_joints; i++)
    {
      float params[3];
      if (fread(params, sizeof(float), 3, jip_file) != 3)
        {
          fprintf(stderr, "Error reading parameters\n");
          goto read_jip_error;
        }

      jip->joint_params[i].bandwidth = params[0];
      jip->joint_params[i].threshold = params[1];
      jip->joint_params[i].offset = params[2];
    }

  if (fclose(jip_file) != 0)
    {
      fprintf(stderr, "Error closing JIP file\n");
    }

  return jip;

read_jip_error:
  free_jip(jip);
  fclose(jip_file);
  return NULL;
}

void
free_jip(JIParams* jip)
{
  if (jip->joint_params)
    {
      xfree(jip->joint_params);
    }
  xfree(jip);
}
