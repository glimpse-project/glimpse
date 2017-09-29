
#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void* xmalloc(size_t size);
void* xaligned_alloc(size_t alignment, size_t size);
void xfree(void *ptr);
void* xcalloc(size_t nmemb, size_t size);
void* xrealloc(void *ptr, size_t size);
void xasprintf(char **strp, const char *fmt, ...);

#ifdef __cplusplus
};
#endif
