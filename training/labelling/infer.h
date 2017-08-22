
#ifndef __INFER__
#define __INFER__

#include <stdint.h>
#include "loader.h"

float* infer(RDTree** forest,
             uint8_t  n_trees,
             float*   depth_image,
             uint32_t width,
             uint32_t height);

#endif /* __INFER__ */
