#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <string.h>

#include "loader.h"
#include "xalloc.h"

RDTree*
load_tree(uint8_t* tree_buf, unsigned len)
{
  RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

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
read_forest(char **files, unsigned n_files)
{
  unsigned n_trees = n_files;
  FILE* tree_fp[n_trees];
  uint8_t* tree_bufs[n_trees];
  unsigned tree_buf_lengths[n_trees];

  for (unsigned i = 0; i < n_files; i++)
    {
        const char* tree_file = files[i];
        struct stat sb;

        tree_fp[i] = fopen(tree_file, "r");
        if (fstat(fileno(tree_fp[i]), &sb) < 0) {
            fprintf(stderr, "Failed to open decision tree: %s\n", tree_file);
            exit(1);
        }
        tree_bufs[i] = (uint8_t*)xcalloc(1, sb.st_size);
        if (fread(tree_bufs[i], sb.st_size, 1, tree_fp[i]) != 1) {
            fprintf(stderr, "Failed to read decision tree: %s\n", tree_file);
            exit(1);
        }
        tree_buf_lengths[i] = sb.st_size;
    }

  return load_forest(tree_bufs, tree_buf_lengths, n_trees);
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