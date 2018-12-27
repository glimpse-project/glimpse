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

#if defined(__APPLE__)
#include <TargetConditionals.h>
#else
#define TARGET_OS_MAC 0
#define TARGET_OS_IOS 0
#define TARGET_OS_OSX 0
#endif

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "xalloc.h"

#define return_if_valid(x) ({ if (x == NULL) abort(); return x; })

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

#ifdef _WIN32
  mem = _aligned_malloc(size, alignment);
#else
  posix_memalign(&mem, alignment, size);
#endif

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
xvasprintf(char **strp, const char *fmt, va_list ap)
{
    if (!strp) {
        fprintf(stderr, "NULL xvasprintf dest ptr, for string: '");
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "'\n");
        abort();
    }

#if defined(__linux__)
    if (vasprintf(strp, fmt, ap) < 0)
        abort();
#else
    va_list len_ap;
    va_copy(len_ap, ap);
    int len = vsnprintf(NULL, 0, fmt, len_ap);
    if (len < 0)
        abort();

    char *str = xmalloc(len + 1);
    va_list copy_ap;
    va_copy(copy_ap, ap);
    vsnprintf(str, len + 1, fmt, copy_ap);
    *strp = str;
#endif
}

void
xasprintf(char **strp, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    xvasprintf(strp, fmt, ap);
    va_end(ap);
}


