#pragma once

#include <stdint.h>

#include "half.hpp"
#include "loader.h"

float* infer(RDTree** forest,
             uint8_t n_trees,
             half_float::half* depth_image,
             uint32_t width,
             uint32_t height);
