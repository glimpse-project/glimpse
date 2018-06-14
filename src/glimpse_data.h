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

#include "half.hpp"

#include "parson.h"

struct gm_data_index;

struct gm_data_index*
gm_data_index_open(struct gm_logger* log,
                   const char* top_dir,
                   const char* index_name,
                   char **err);

void
gm_data_index_destroy(struct gm_data_index* data_index);

int
gm_data_index_get_len(struct gm_data_index* data_index);

const char*
gm_data_index_get_top_dir(struct gm_data_index* data_index);

JSON_Value*
gm_data_index_get_meta(struct gm_data_index* data_index);

int
gm_data_index_get_width(struct gm_data_index* data_index);

int
gm_data_index_get_height(struct gm_data_index* data_index);

int
gm_data_index_get_n_labels(struct gm_data_index* data_index);

const char *
gm_data_index_get_frame_path(struct gm_data_index* data_index, int n);

bool
gm_data_index_foreach(struct gm_data_index* data_index,
                      bool (*callback)(struct gm_data_index* data_index,
                                       int index,
                                       const char* frame_path,
                                       void* user_data,
                                       char** err),
                      void* user_data,
                      char** err);

bool
gm_data_index_load_joints(struct gm_data_index* data_index,
                          const char* joint_map_file,
                          int* out_n_joints,
                          float** out_joints,
                          char** err);

JSON_Value*
gm_data_load_simple(struct gm_logger *log,
                    const char* data_dir,
                    const char* index_name,
                    const char* joint_map_path,
                    int* out_n_images,
                    int* out_n_joints,
                    int* out_width,
                    int* out_height,
                    half_float::half** out_depth_images,
                    uint8_t** out_label_images,
                    float** out_joints,
                    char** err);

JSON_Value*
gm_data_load_label_map_from_json(struct gm_logger* log,
                                 const char* filename,
                                 uint8_t* map,
                                 char** err);

