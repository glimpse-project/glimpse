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

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <vector>

#include "glimpse_log.h"
#include "glimpse_mem_pool.h"


struct gm_mem_pool {
    struct gm_logger *log;

    char *name;

    pthread_mutex_t lock;
    pthread_cond_t available_cond;
    unsigned max_size;
    std::vector<void *> available;
    std::vector<void *> busy;

    void *(*alloc_mem)(struct gm_mem_pool *pool, void *user_data);
    void (*free_mem)(struct gm_mem_pool *pool, void *mem, void *user_data);
    void *user_data;
};

struct gm_mem_pool *
mem_pool_alloc(struct gm_logger *log,
               const char *name,
               unsigned max_size,
               void *(*alloc_mem)(struct gm_mem_pool *pool, void *user_data),
               void (*free_mem)(struct gm_mem_pool *pool, void *mem,
                                void *user_data),
               void *user_data)
{
    struct gm_mem_pool *pool = new gm_mem_pool();

    pool->log = log;
    pool->max_size = max_size;
    pool->name = strdup(name);
    pool->alloc_mem = alloc_mem;
    pool->free_mem = free_mem;
    pool->user_data = user_data;

    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->available_cond, NULL);

    return pool;
}

void
mem_pool_free(struct gm_mem_pool *pool)
{
    mem_pool_free_resources(pool);
    free(pool->name);
    free(pool);
}

void *
mem_pool_acquire_resource(struct gm_mem_pool *pool)
{
    void *resource;

    pthread_mutex_lock(&pool->lock);

    /* Sanity check with arbitrary upper limit for the number of allocations */
    gm_assert(pool->log,
              (pool->busy.size() + pool->available.size()) < 100,
              "'%s' memory pool growing out of control (%lu allocations)",
              pool->name,
              (pool->busy.size() + pool->available.size()));

    if (pool->available.size()) {
        resource = pool->available.back();
        pool->available.pop_back();
    } else if (pool->busy.size() + pool->available.size() > pool->max_size) {

        gm_debug(pool->log,
                 "Throttling \"%s\" pool acquisition, waiting for old %s object to be released\n",
                 pool->name, pool->name);

        while (!pool->available.size())
            pthread_cond_wait(&pool->available_cond, &pool->lock);

        resource = pool->available.back();
        pool->available.pop_back();
    } else {
        resource = pool->alloc_mem(pool, pool->user_data);
    }

    pool->busy.push_back(resource);

    pthread_mutex_unlock(&pool->lock);

    return resource;
}

void
mem_pool_recycle_resource(struct gm_mem_pool *pool, void *resource)
{
    pthread_mutex_lock(&pool->lock);

    unsigned size = pool->busy.size();
    for (unsigned i = 0; i < size; i++) {
        if (pool->busy[i] == resource) {
            pool->busy[i] = pool->busy.back();
            pool->busy.pop_back();
            break;
        }
    }

    gm_assert(pool->log,
              pool->busy.size() == (size - 1),
              "Didn't find recycled resource %p in %s pool's busy list",
              resource,
              pool->name);

    pool->available.push_back(resource);
    pthread_cond_broadcast(&pool->available_cond);
    pthread_mutex_unlock(&pool->lock);
}

void
mem_pool_free_resources(struct gm_mem_pool *pool)
{
    gm_assert(pool->log,
              pool->busy.size() == 0,
              "Shouldn't be freeing a pool with resources still in use");

    while (pool->available.size()) {
        void *resource = pool->available.back();
        pool->available.pop_back();
        pool->free_mem(pool, resource, pool->user_data);
    }
}


