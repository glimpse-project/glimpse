
#ifndef __LOADER__
#define __LOADER__

#include <stdio.h>
#include "utils.h"

typedef struct {
  RDTHeader header;
  Node* nodes;
  float* label_pr_tables;
} RDTree;

RDTree* read_tree(FILE* file);
void free_tree(RDTree* tree);

RDTree** read_forest(char** files, uint8_t n_files);
void free_forest(RDTree** forest, uint8_t n_trees);

#endif /* __LOADER */
