#pragma once

#include <stdint.h>

#include "half.hpp"
#include "loader.h"
#include "llist.h"
#include "parson.h"

#define HUGE_DEPTH 1000.f

float* infer_labels(RDTree** forest,
                    uint8_t n_trees,
                    half_float::half* depth_image,
                    uint32_t width,
                    uint32_t height,
                    float* out_labels = NULL);

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
                    JIParam* params,
                    float* out_joints = NULL);

float* reproject(half_float::half* depth_image,
                 int32_t width,
                 int32_t height,
                 float vfov,
                 float threshold,
                 uint32_t* n_points,
                 float* out_points = NULL);

half_float::half* project(float* point_cloud,
                          uint32_t n_points,
                          int32_t width,
                          int32_t height,
                          float vfov,
                          float background = 0.f,
                          half_float::half* out_depth = NULL);
