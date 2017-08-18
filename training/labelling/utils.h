
#ifndef __UTILS__
#define __UTILS__

#include <stdint.h>
#include <sys/types.h>
#include <math.h>

#define OUT_VERSION 3

#define vector(type,size) type __attribute__ ((vector_size(sizeof(type)*(size))))

typedef vector(int32_t, 2) Int2D;
typedef vector(float, 4) UVPair;

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
  float   fov;
} RDTHeader;

inline float
sample_uv(float* depth_image, uint32_t width, uint32_t height,
          Int2D pixel, float depth, UVPair uv)
{
#if 0
  // This code path is slower. gcc is cleverer than me, leaving this here as
  // a reminder.
  vector(float, 4) uv_pixel = { (float)pixel[0], (float)pixel[1],
                                (float)pixel[0], (float)pixel[1] };
  uv_pixel += uv / depth;

  vector(float, 4) extents = { (float)width, (float)height,
                               (float)width, (float)height };
  vector(int, 4) mask = (uv_pixel >= 0.f && uv_pixel < extents);

  float upixel = (mask[0] && mask[1]) ?
    depth_image[(((uint32_t)uv_pixel[1] * width) + (uint32_t)uv_pixel[0])] : INFINITY;
  float vpixel = (mask[2] && mask[3]) ?
    depth_image[(((uint32_t)uv_pixel[3] * width) + (uint32_t)uv_pixel[2])] : INFINITY;

  return upixel - vpixel;
#else
  Int2D u = { (int32_t)(pixel[0] + uv[0] / depth),
              (int32_t)(pixel[1] + uv[1] / depth) };
  Int2D v = { (int32_t)(pixel[0] + uv[2] / depth),
              (int32_t)(pixel[1] + uv[3] / depth) };

  float upixel = (u[0] >= 0 && u[0] < (int32_t)width &&
                  u[1] >= 0 && u[1] < (int32_t)height) ?
    depth_image[((u[1] * width) + u[0])] : INFINITY;
  float vpixel = (v[0] >= 0 && v[0] < (int32_t)width &&
                  v[1] >= 0 && v[1] < (int32_t)height) ?
    depth_image[((v[1] * width) + v[0])] : INFINITY;

  return upixel - vpixel;
#endif
}

#endif /* __UTILS__ */

