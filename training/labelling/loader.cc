#include <stdio.h>
#include <string.h>

#include "loader.h"
#include "xalloc.h"

RDTree*
read_tree(FILE* file)
{
  RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

  if (fread(&tree->header, sizeof(RDTHeader), 1, file) != 1)
    {
      fprintf(stderr, "Error reading header\n");
      free_tree(tree);
      return NULL;
    }

  if (strncmp(tree->header.tag, "RDT", 3) != 0)
    {
      fprintf(stderr, "File is not an RDT file\n");
      free_tree(tree);
      return NULL;
    }

  if (tree->header.version != OUT_VERSION)
    {
      fprintf(stderr, "Incompatible RDT version, expected %u, found %u\n",
              OUT_VERSION, (uint32_t)tree->header.version);
      free_tree(tree);
      return NULL;
    }

  // Read in the decision tree nodes
  uint32_t n_nodes = (uint32_t)roundf(powf(2.f, tree->header.depth)) - 1;
  tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
  if (fread(tree->nodes, sizeof(Node), n_nodes, file) != n_nodes)
    {
      fprintf(stderr, "Error reading tree nodes\n");
      free_tree(tree);
      return NULL;
    }

  // Read in the label probabilities
  long label_pr_pos = ftell(file);
  fseek(file, 0, SEEK_END);
  long label_bytes = ftell(file) - label_pr_pos;
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
  fseek(file, label_pr_pos, SEEK_SET);

  tree->label_pr_tables = (float*)xmalloc(label_bytes);
  if (fread(tree->label_pr_tables, sizeof(float) * tree->header.n_labels,
            n_tables, file) != n_tables)
    {
      fprintf(stderr, "Error reading label probability tables\n");
      free_tree(tree);
      return NULL;
    }

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
read_forest(char** files, uint8_t n_files)
{
  bool error = false;
  uint8_t n_labels = 0;
  RDTree** trees = (RDTree**)xcalloc(n_files, sizeof(RDTree*));

  for (uint8_t i = 0; i < n_files; i++)
    {
      // Validate the decision tree
      FILE* tree_file = fopen(files[i], "rb");
      if (!tree_file)
        {
          fprintf(stderr, "Error opening tree '%s'\n", files[i]);
          error = true;
          break;
        }
      RDTree* tree = trees[i] = read_tree(tree_file);
      fclose(tree_file);

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
          fprintf(stderr, "Tree in '%s' has %u labels, expected %u\n",
                  files[i], (uint32_t)tree->header.n_labels,
                  (uint32_t)n_labels);
          error = true;
          break;
        }
    }

  if (error)
    {
      for (uint8_t i = 0; i < n_files; i++)
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

void
free_forest(RDTree** forest, uint8_t n_trees)
{
  for (uint8_t i = 0; i < n_trees; i++)
    {
      free_tree(forest[i]);
    }
  xfree(forest);
}
