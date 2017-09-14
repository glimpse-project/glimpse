
#include <stdarg.h>
#include <stdio.h>

#include "xalloc.h"

#define return_if_valid(x) if (x == NULL) exit(1); return x

void*
xmalloc(size_t size)
{
  void* mem = malloc(size);
  return_if_valid(mem);
}

void*
xaligned_alloc(size_t alignment, size_t size)
{
  void* mem = aligned_alloc(alignment, size);
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

void
xasprintf(char **strp, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);

    if (!fmt) {
        vfprintf(stderr, fmt, ap);
        exit(1);
    } else {
        if (vasprintf(strp, fmt, ap) < 0)
            exit(1);
    }

    va_end(ap);
}


