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

typedef struct _LList LList;

struct _LList {
  LList* prev;
  LList* next;
  void* data;
};

typedef bool (*LListIterCallback)(LList*   node,
                                  uint32_t index,
                                  void*    userdata);

typedef int (*LListSearchCallback)(LList*   a,
                                   LList*   b,
                                   void*    userdata);

#ifdef __cplusplus
extern "C" {
#endif

LList*   llist_new(void* data);

void     llist_foreach(LList*            list,
                       LListIterCallback cb,
                       void*             userdata);

bool     llist_free_cb(LList*   node,
                       uint32_t index,
                       void*    userdata);

void     llist_free(LList*            list,
                    LListIterCallback free_cb,
                    void*             userdata);

LList*   llist_insert_before(LList* before,
                             LList* node);

LList*   llist_insert_after(LList* after,
                            LList* node);

LList*   llist_remove(LList* node);

LList*   llist_reverse(LList* node);

LList*   llist_first(LList* node);

LList*   llist_last(LList* node);

uint32_t llist_length(LList* node);

void**   llist_as_array(LList*    node,
                        uint32_t* length);

LList*   llist_from_array(void**   data,
                          uint32_t length);

LList*   llist_shuffle(LList* node);

LList*   llist_slice(LList*            node,
                     uint32_t          begin,
                     uint32_t          end,
                     LListIterCallback free_cb,
                     void*             userdata);

void*    llist_pop(LList**           node,
                   LListIterCallback free_cb,
                   void*             userdata);

LList*   llist_sort(LList*              node,
                    LListSearchCallback sort_cb,
                    void*               userdata);

#ifdef __cplusplus
};
#endif
