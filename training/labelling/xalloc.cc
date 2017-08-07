
#include "xalloc.h"

#define return_if_valid(x) if (x == NULL) exit(1); return x

void*
xmalloc(size_t size)
{
  void* mem = malloc(size);
  return_if_valid(mem);
}

void
xfree(void *ptr)
{
  free(ptr);
}

void*
xcalloc(size_t nmemb, size_t size)
{
  void* mem = calloc(nmemb, size);
  return_if_valid(mem);
}

void *
xrealloc(void *ptr, size_t size)
{
  if (size == 0) {
    free(ptr);
    return NULL;
  }
  ptr = realloc(ptr, size);
  return_if_valid(ptr);
}
