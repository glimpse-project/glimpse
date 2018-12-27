#pragma once

#ifdef __unix__

#include <stdio.h>

#else

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Defined with intptr_t instead of ssize_t which isn't defined on Windows
intptr_t getline(char **lineptr, size_t *n, FILE *stream);

#ifdef __cplusplus
}
#endif

#endif // __unix__
