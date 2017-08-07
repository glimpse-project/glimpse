
#ifndef __UTILS__
#define __UTILS__

#include <stdint.h>
#include <sys/types.h>

#define OUT_VERSION 1

typedef struct {
  int32_t x;
  int32_t y;
} Int2D;

typedef struct {
  float x;
  float y;
} Float2D;

typedef struct {
  Float2D u;
  Float2D v;
} UVPair;

typedef struct {
  UVPair uv;              // U and V parameters
  float t;                // Threshold
  uint32_t label_pr_idx;  // Index into label probability table (1-based)
} Node;

typedef struct __attribute__((__packed__)) {
  char    tag[3];
  uint8_t version;
  uint8_t depth;
  uint8_t n_labels;
} RDLHeader;

float
sample_uv(float* depth_image,
          uint32_t width,
          uint32_t height,
          Int2D* pixel,
          float depth,
          UVPair* uv);

#endif /* __UTILS__ */

