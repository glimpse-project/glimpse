
#pragma once

#include <stdio.h>

#include "llist.h"

#define vector(type,size) type __attribute__ ((vector_size(sizeof(type)*(size))))

#define RDT_VERSION 3

typedef struct {
  vector(float,4) uv;     // U in [0:2] and V in [2:4]
  float t;                // Threshold
  uint32_t label_pr_idx;  // Index into label probability table (1-based)
} Node;

typedef struct __attribute__((__packed__)) {
  char    tag[3];
  uint8_t version;
  uint8_t depth;
  uint8_t n_labels;
  float   fov;
} RDTHeader;

typedef struct {
  RDTHeader header;
  Node* nodes;
  float* label_pr_tables;
} RDTree;

typedef struct {
  float bandwidth;
  float threshold;
  float offset;
} JIParam;

typedef struct __attribute__((__packed__)) {
  char    tag[3];
  uint8_t version;
  uint8_t n_joints;
} JIPHeader;

typedef struct {
  JIPHeader header;
  JIParam*  joint_params;
} JIParams;

#ifdef __cplusplus
extern "C" {
#endif

RDTree* load_tree(uint8_t* tree, unsigned len);
void free_tree(RDTree* tree);

RDTree** read_forest(char** files, unsigned n_files);
RDTree** load_forest(uint8_t** tree_bufs, unsigned* tree_buf_lengths, unsigned n_trees);
void free_forest(RDTree** forest, int n_trees);

LList** read_jointmap(char* filename, uint8_t n_joints, char*** joint_names);
void free_jointmap(LList** jointmap, uint8_t n_joints, char** joint_names);

JIParams* read_jip(char* filename);
void free_jip(JIParams* jip);

#ifdef __cplusplus
};
#endif
