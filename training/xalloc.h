
#ifndef __XALLOC__
#define __XALLOC__

#include <stdlib.h>

void* xmalloc(size_t size);
void* xaligned_alloc(size_t alignment, size_t size);
void xfree(void *ptr);
void* xcalloc(size_t nmemb, size_t size);
void* xrealloc(void *ptr, size_t size);
void xasprintf(char **strp, const char *fmt, ...);

#endif /* __XALLOC__ */
