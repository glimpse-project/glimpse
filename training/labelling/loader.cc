#include <stdio.h>
#include <string.h>

#include "loader.h"
#include "xalloc.h"

RDTree*
read_rdt(FILE* file)
{
  RDTree* tree = (RDTree*)xcalloc(1, sizeof(RDTree));

  if (fread(&tree->header, sizeof(RDTHeader), 1, file) != 1)
    {
      fprintf(stderr, "Error reading header\n");
      free_rdt(tree);
      return NULL;
    }

  if (strncmp(tree->header.tag, "RDT", 3) != 0)
    {
      fprintf(stderr, "File is not an RDT file\n");
      free_rdt(tree);
      return NULL;
    }

  if (tree->header.version != OUT_VERSION)
    {
      fprintf(stderr, "Incompatible RDT version, expected %u, found %u\n",
              OUT_VERSION, (uint32_t)tree->header.version);
      free_rdt(tree);
      return NULL;
    }

  // Read in the decision tree nodes
  uint32_t n_nodes = (uint32_t)roundf(powf(2.f, tree->header.depth)) - 1;
  tree->nodes = (Node*)xmalloc(n_nodes * sizeof(Node));
  if (fread(tree->nodes, sizeof(Node), n_nodes, file) != n_nodes)
    {
      fprintf(stderr, "Error reading tree nodes\n");
      free_rdt(tree);
      return NULL;
    }

  // Read in the label probabilities
  long label_pr_pos = ftell(file);
  fseek(file, 0, SEEK_END);
  long label_bytes = ftell(file) - label_pr_pos;
  if (label_bytes % sizeof(float) != 0)
    {
      fprintf(stderr, "Unexpected size of label probability tables\n");
      free_rdt(tree);
      return NULL;
    }
  uint32_t n_prs = label_bytes / sizeof(float);
  if (n_prs % tree->header.n_labels != 0)
    {
      fprintf(stderr, "Unexpected number of label probabilities\n");
      free_rdt(tree);
      return NULL;
    }
  uint32_t n_tables = n_prs / tree->header.n_labels;
  fseek(file, label_pr_pos, SEEK_SET);

  tree->label_pr_tables = (float*)xmalloc(label_bytes);
  if (fread(tree->label_pr_tables, sizeof(float) * tree->header.n_labels,
            n_tables, file) != n_tables)
    {
      fprintf(stderr, "Error reading label probability tables\n");
      free_rdt(tree);
      return NULL;
    }

  return tree;
}

void
free_rdt(RDTree* tree)
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
