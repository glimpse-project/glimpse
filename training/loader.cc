#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cstddef>

#include "loader.h"
#include "parson.h"
#include "xalloc.h"

RDTree*
load_tree(uint8_t* tree_buf, unsigned len)
{
  RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

  static_assert(sizeof(RDTHeader) == 10, "RDT ABI Breakage");
  static_assert(sizeof(Node) == 32,      "RDT ABI Breakage");
  static_assert(offsetof(Node, t) == 16, "RDT ABI Breakage");

  if (len < sizeof(RDTHeader)) {
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
              RDT_VERSION, (uint32_t)tree->header.version);
      free_tree(tree);
      return NULL;
    }

  // Read in the decision tree nodes
  uint32_t n_nodes = (uint32_t)roundf(powf(2.f, tree->header.depth)) - 1;
  tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
  if (len < (sizeof(Node) * n_nodes)) {
      fprintf(stderr, "Error parsing tree nodes\n");
      free_tree(tree);
      return NULL;
  }
  memcpy(tree->nodes, tree_buf, sizeof(Node) * n_nodes);
  tree_buf += sizeof(Node) * n_nodes;
  len -= sizeof(Node) * n_nodes;

  // Read in the label probabilities
  long label_bytes = len;
  if (label_bytes % sizeof(float) != 0)
    {
      fprintf(stderr, "Unexpected size of label probability tables\n");
      free_tree(tree);
      return NULL;
    }
  uint32_t n_prs = label_bytes / sizeof(float);
  if (n_prs % tree->header.n_labels != 0)
    {
      fprintf(stderr, "Unexpected number of label probabilities\n");
      free_tree(tree);
      return NULL;
    }
  uint32_t n_tables = n_prs / tree->header.n_labels;

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

RDTree**
load_forest(uint8_t** tree_bufs, unsigned* tree_buf_lengths, unsigned n_trees)
{
  bool error = false;
  uint8_t n_labels = 0;
  RDTree** trees = (RDTree**)xcalloc(n_trees, sizeof(RDTree*));

  for (unsigned i = 0; i < n_trees; i++)
    {
      // Validate the decision tree
      RDTree* tree = trees[i] = load_tree(tree_bufs[i], tree_buf_lengths[i]);
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
          fprintf(stderr, "Tree %u has %u labels, expected %u\n",
                  i, (uint32_t)tree->header.n_labels,
                  (uint32_t)n_labels);
          error = true;
          break;
        }
    }

  if (error)
    {
      for (unsigned i = 0; i < n_trees; i++)
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

RDTree **
read_forest(const char **files, unsigned n_files)
{
  uint8_t* tree_bufs[n_files];
  unsigned tree_buf_lengths[n_files];
  unsigned n_trees = 0;
  bool error = false;

  for (unsigned i = 0; i < n_files; i++)
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
    NULL : load_forest(tree_bufs, tree_buf_lengths, n_trees);

  for (unsigned i = 0; i < n_trees; i++)
    {
      xfree(tree_bufs[i]);
    }

  return forest;
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
