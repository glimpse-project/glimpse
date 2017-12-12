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


#include <stdlib.h>
#include "llist.h"
#include "xalloc.h"

LList*
llist_new(void* data)
{
  LList* list = (LList*)xcalloc(1, sizeof(LList));
  list->data = data;
  return list;
}

void
llist_foreach(LList* list, LListIterCallback cb, void* userdata)
{
  uint32_t i = 0;
  while (list != NULL)
    {
      cb(list, i++, userdata);
      list = list->next;
    }
}

bool
llist_free_cb(LList* node, uint32_t index, void* userdata)
{
  xfree(node->data);
  return true;
}

void
llist_free(LList* list, LListIterCallback free_cb, void* userdata)
{
  uint32_t i = 0;
  while (list != NULL)
    {
      LList* next = list->next;
      if (free_cb)
        {
          free_cb(list, i++, userdata);
        }
      xfree(list);
      list = next;
    }
}

LList*
llist_insert_before(LList* before, LList* node)
{
  if (before)
    {
      if (before->prev)
        {
          before->prev->next = node;
          node->prev = before->prev;
        }
      before->prev = node;
    }
  node->next = before;

  return node;
}

LList*
llist_insert_after(LList* after, LList* node)
{
  if (after)
    {
      if (after->next)
        {
          after->next->prev = node;
          node->next = after->next;
        }
      after->next = node;
    }
  node->prev = after;

  return node;
}

LList*
llist_remove(LList* node)
{
  if (node->prev)
    {
      node->prev->next = node->next;
    }
  if (node->next)
    {
      node->next->prev = node->prev;
    }
  node->prev = NULL;
  node->next = NULL;
  return node;
}

LList*
llist_reverse(LList* node)
{
  node = llist_first(node);

  while (node != NULL)
    {
      LList* tmp = node->prev;
      node->prev = node->next;
      node->next = tmp;
      node = node->prev;
    }

  return node;
}

LList*
llist_first(LList* node)
{
  if (node)
    {
      while (node->prev != NULL) node = node->prev;
    }
  return node;
}

LList*
llist_last(LList* node)
{
  if (node)
    {
      while (node->next != NULL) node = node->next;
    }
  return node;
}

uint32_t
llist_length(LList* node)
{
  uint32_t length = 0;
  while (node != NULL)
    {
      ++length;
      node = node->next;
    }
  return length;
}

void**
llist_as_array(LList* node, uint32_t* out_length)
{
  uint32_t length = llist_length(node);
  void** array = (void**)xmalloc(length * sizeof(void*));
  for (int i = 0; node; node = node->next, i++)
    {
      array[i] = node->data;
    }
  if (out_length)
    {
      *out_length = length;
    }
  return array;
}

LList*
llist_from_array(void** data, uint32_t length)
{
  if (length == 0)
    {
      return NULL;
    }

  LList* first = llist_new(data[0]);
  LList* last = first;
  for (uint32_t i = 1; i < length; i++)
    {
      last = llist_insert_after(last, llist_new(data[i]));
    }

  return first;
}

LList*
llist_shuffle(LList* node)
{
  uint32_t length;
  void** array = llist_as_array(node, &length);
  llist_free(node, NULL, NULL);

  for (uint32_t i = length; i > 0; --i)
    {
      // We're not too worried about how random this is
      uint32_t j = rand() % i;
      void* tmp = array[i - 1];
      array[i - 1] = array[j];
      array[j] = tmp;
    }

  node = llist_from_array(array, length);
  xfree(array);
  return node;
}

LList*
llist_slice(LList* node, uint32_t begin, uint32_t end,
            LListIterCallback free_cb, void* userdata)
{
  LList* sliced = NULL;
  for (uint32_t i = 0; i < end && node; i++)
    {
      LList* next = node->next;
      if (i == begin)
        {
          sliced = node;
        }
      else if (i == end - 1)
        {
          node->next = NULL;
        }
      else if (i < begin || i > end)
        {
          if (free_cb)
            {
              free_cb(node, i, userdata);
            }
          xfree(node);
        }
      node = next;
    }
  return sliced;
}

void*
llist_pop(LList** node, LListIterCallback cb, void* userdata)
{
  if (!node || !(*node))
    {
      return NULL;
    }

  void* data = (*node)->data;
  LList* remove = *node;
  *node = (*node)->next;
  llist_free(llist_remove(remove), cb, userdata);

  return data;
}

LList*
llist_sort(LList* node, LListSearchCallback sort_cb, void* userdata)
{
  for (LList* last = llist_last(node); last; last = last->prev)
    {
      for (LList* i = node; i != last; i = i->next)
        {
          if (sort_cb(i, i->next, userdata) < 0)
            {
              void* tmp = i->next->data;
              i->next->data = i->data;
              i->data = tmp;
            }
        }
    }

  return node;
}
