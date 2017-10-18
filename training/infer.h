#pragma once

#include <stdint.h>

#include "half.hpp"
#include "loader.h"
#include "llist.h"
#include "parson.h"

float* infer_labels(RDTree** forest,
                    uint8_t n_trees,
                    half_float::half* depth_image,
                    uint32_t width,
                    uint32_t height);

float* calc_pixel_weights(half_float::half* depth_image,
                          float* pr_table,
                          int32_t width,
                          int32_t height,
                          uint8_t n_labels,
                          JSON_Value* joint_map,
                          float* out_weights = NULL);

float* infer_joints(half_float::half* depth_image,
                    float* pr_table,
                    float* weights,
                    int32_t width,
                    int32_t height,
                    uint8_t n_labels,
                    JSON_Value* joint_map,
                    float vfov,
                    JIParam* params);
