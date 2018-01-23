/*
 * Copyright (C) 2017 Glimp IP Ltd
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <stdint.h>

#include "loader.h"
#include "llist.h"
#include "parson.h"

#define HUGE_DEPTH 1000.f

typedef struct {
  float x;
  float y;
  float z;
  float confidence;
} Joint;

typedef struct {
  int     n_joints;
  LList** joints;
} InferredJoints;

template<typename FloatT>
float* infer_labels(RDTree** forest,
                    uint8_t n_trees,
                    FloatT* depth_image,
                    uint32_t width,
                    uint32_t height,
                    float* out_labels = NULL);

template<typename FloatT>
float* calc_pixel_weights(FloatT* depth_image,
                          float* pr_table,
                          int32_t width,
                          int32_t height,
                          uint8_t n_labels,
                          JSON_Value* joint_map,
                          float* out_weights = NULL);

template<typename FloatT>
InferredJoints* infer_joints_fast(FloatT* depth_image,
                                  float* pr_table,
                                  float* weights,
                                  int32_t width,
                                  int32_t height,
                                  uint8_t n_labels,
                                  JSON_Value* joint_map,
                                  float vfov,
                                  JIParam* params);

template<typename FloatT>
InferredJoints* infer_joints(FloatT* depth_image,
                             float* pr_table,
                             float* weights,
                             int32_t width,
                             int32_t height,
                             uint8_t n_labels,
                             JSON_Value* joint_map,
                             float vfov,
                             JIParam* params);

void free_joints(InferredJoints* joints);

template<typename FloatT>
float* reproject(FloatT* depth_image,
                 int32_t width,
                 int32_t height,
                 float vfov,
                 float threshold,
                 uint32_t* n_points,
                 float* out_points = NULL);

template<typename FloatT>
FloatT* project(float* point_cloud,
                uint32_t n_points,
                int32_t width,
                int32_t height,
                float vfov,
                float background = 0.f,
                FloatT* out_depth = NULL);

