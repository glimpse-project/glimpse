#pragma once

#ifndef _WIN32
#include <libgen.h>
#else

#ifdef __cplusplus
extern "C" {
#endif
char *basename(char *path);
char *dirname(char *path);
#ifdef __cplusplus
}
#endif

#endif
