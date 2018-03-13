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

#define _GNU_SOURCE         // vasprintf
#define _XOPEN_SOURCE 600   // posix_memalign

#include <stdlib.h>
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
  void* mem = NULL;

  posix_memalign(&mem, alignment, size);

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

    if (!strp) {
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
        exit(1);
    } else {
#ifdef __linux__
        va_start(ap, fmt);
        if (vasprintf(strp, fmt, ap) < 0)
            exit(1);
        va_end(ap);
#else
        va_start(ap, fmt);
        int len = vsnprintf(NULL, 0, fmt, ap);
        if (len < 0)
            exit(1);
        va_end(ap);
        va_start(ap, fmt);
        char *str = xmalloc(len + 1);
        vsnprintf(str, len + 1, fmt, ap);
        va_end(ap);
        *strp = str;
#endif
    }

}


