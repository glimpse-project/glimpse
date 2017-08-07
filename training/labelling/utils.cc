#include <math.h>
#include "utils.h"

float
sample_uv(float* depth_image, uint32_t width, uint32_t height,
          Int2D* pixel, float depth, UVPair* uv)
{
  Int2D u = { (int32_t)(pixel->x + uv->u.x / depth),
              (int32_t)(pixel->y + uv->u.y / depth) };
  Int2D v = { (int32_t)(pixel->x + uv->v.x / depth),
              (int32_t)(pixel->y + uv->v.y / depth) };

  float upixel = (u.x >= 0 && u.x < (int32_t)width &&
                  u.y >= 0 && u.y < (int32_t)height) ?
    depth_image[((u.y * width) + u.x)] : INFINITY;
  float vpixel = (v.x >= 0 && v.x < (int32_t)width &&
                  v.y >= 0 && v.y < (int32_t)height) ?
    depth_image[((v.y * width) + v.x)] : INFINITY;

  return upixel - vpixel;
}
