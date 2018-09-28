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
#include <stdbool.h>

#include "jip.h"
#include "llist.h"
#include "parson.h"

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


struct joints_inferrer;

#ifdef __cplusplus
extern "C" {
#endif

struct joints_inferrer*
joints_inferrer_new(struct gm_logger* logger,
                    JSON_Value* joint_map,
                    char** err);

void
joints_inferrer_destroy(struct joints_inferrer* inferrer);

float*
joints_inferrer_calc_pixel_weights(struct joints_inferrer* inferrer,
                                   float* depth_image,
                                   float* pr_table,
                                   int width,
                                   int height,
                                   int n_labels,
                                   float* out_weights);

InferredJoints*
joints_inferrer_infer_fast(struct joints_inferrer* inferrer,
                           float* depth_image,
                           float* pr_table,
                           float* weights,
                           int width,
                           int height,
                           int n_labels,
                           float vfov,
                           JIParam* params);

InferredJoints*
joints_inferrer_infer(struct joints_inferrer* inferrer,
                      float* depth_image,
                      float* pr_table,
                      float* weights,
                      int width,
                      int height,
                      float bg_depth,
                      int n_labels,
                      float vfov,
                      JIParam* params);

void
joints_inferrer_free_joints(struct joints_inferrer* inferrer,
                            InferredJoints* joints);

#ifdef __cplusplus
}
#endif
