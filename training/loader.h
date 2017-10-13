
#ifndef __LOADER__
#define __LOADER__

#include <stdio.h>
#include "utils.h"
#include "llist.h"

typedef struct {
  RDTHeader header;
  Node* nodes;
  float* label_pr_tables;
} RDTree;

RDTree* load_tree(uint8_t* tree, unsigned len);
void free_tree(RDTree* tree);

RDTree** read_forest(char** files, unsigned n_files);
RDTree** load_forest(uint8_t** tree_bufs, unsigned* tree_buf_lengths, unsigned n_trees);
void free_forest(RDTree** forest, int n_trees);

LList** read_jointmap(char* filename, uint8_t n_joints, char*** joint_names);
void free_jointmap(LList** jointmap, uint8_t n_joints, char** joint_names);

#endif /* __LOADER */
