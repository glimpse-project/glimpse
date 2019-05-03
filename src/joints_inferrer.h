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

#include "glimpse_context.h"

typedef struct {
    float x;
    float y;
    float z;
    float confidence;
} Joint;

typedef struct {
    int n_joints;
    LList *joints[];
} InferredJoints;


struct joints_inferrer;
struct joints_inferrer_state;

#ifdef __cplusplus
extern "C" {
#endif

struct joints_inferrer *
joints_inferrer_new(struct gm_logger *logger,
                    JSON_Value *joint_map,
                    int n_labels,
                    char **err);

void
joints_inferrer_destroy(struct joints_inferrer *inferrer);

struct joints_inferrer_state *
joints_inferrer_state_new(struct joints_inferrer *inferrer);

void
joints_inferrer_state_destroy(struct joints_inferrer_state *state);

float *
joints_inferrer_calc_pixel_weights(struct joints_inferrer_state *state,
                                   float *depth_image,
                                   float *pr_table,
                                   int width,
                                   int height,
                                   float *out_weights);

InferredJoints *
joints_inferrer_infer_fast(struct joints_inferrer_state *state,
                           struct gm_intrinsics *intrinsics,
                           int cluster_width,
                           int cluster_height,
                           int cluster_x0,
                           int cluster_y0,
                           float *cluster_depth_image,
                           float *cluster_label_probs,
                           float *cluster_weights,
                           JIParam *params,
                           bool debug);

InferredJoints *
joints_inferrer_infer(struct joints_inferrer_state *state,
                      struct gm_intrinsics *intrinsics,
                      int cluster_width,
                      int cluster_height,
                      int cluster_x0,
                      int cluster_y0,
                      float *cluster_depth_image,
                      float *cluster_label_probs,
                      float *cluster_weights,
                      float bg_depth,
                      JIParam *params);

const Joint *
joints_inferrer_state_get_candidates(struct joints_inferrer_state *state,
                                     int joint,
                                     int *n_candidates_out);

void
joints_inferrer_state_free_joints(struct joints_inferrer_state *state,
                                  InferredJoints *joints);

#ifdef __cplusplus
}
#endif
