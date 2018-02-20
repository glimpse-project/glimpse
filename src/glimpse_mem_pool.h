/*
 * Copyright (C) 2018 Glimp IP Ltd
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

#include <pthread.h>

struct gm_mem_pool;

#ifdef __cplusplus
extern "C" {
#endif

struct gm_mem_pool *
mem_pool_alloc(struct gm_logger *log,
               const char *name,
               unsigned max_size,
               void *(*alloc_mem)(struct gm_mem_pool *pool, void *user_data),
               void (*free_mem)(struct gm_mem_pool *pool, void *mem,
                                void *user_data),
               void *user_data);

void
mem_pool_free(struct gm_mem_pool *pool);

void *
mem_pool_acquire_resource(struct gm_mem_pool *pool);

void
mem_pool_recycle_resource(struct gm_mem_pool *pool, void *resource);

void
mem_pool_free_resources(struct gm_mem_pool *pool);

const char *
mem_pool_get_name(struct gm_mem_pool *pool);

void
mem_pool_foreach(struct gm_mem_pool *pool,
                 void (*callback)(struct gm_mem_pool *pool,
                                  void *resource,
                                  void *user_data),
                 void *user_data);

#ifdef __cplusplus
}
#endif
