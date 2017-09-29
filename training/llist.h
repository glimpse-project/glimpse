
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

#ifdef __cplusplus
extern "C" {
#endif

LList*   llist_new(void* data);

void     llist_foreach(LList*            list,
                       LListIterCallback cb,
                       void*             userdata);

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

#ifdef __cplusplus
};
#endif
