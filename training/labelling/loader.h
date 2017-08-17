
#ifndef __LOADER__
#define __LOADER__

#include <stdio.h>
#include "utils.h"

typedef struct {
  RDTHeader header;
  Node* nodes;
  float* label_pr_tables;
} RDTree;

RDTree* read_rdt(FILE* file);
void free_rdt(RDTree* tree);

#endif /* __LOADER */
