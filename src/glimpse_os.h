#pragma once

#include <stdint.h>

// Convenience for including the right header for alloca...
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <alloca.h>
#endif

#include "glimpse_log.h"

#ifdef __cplusplus
extern "C" {
#endif

uint64_t
gm_os_get_time(void);

bool
gm_os_mkdir(struct gm_logger *log, const char *path, char **err);

#if defined(__unix__) || defined(__APPLE__)

#include <unistd.h>

#define gm_os_usleep(X) usleep(X)

#elif defined(_WIN32)

static inline int
gm_os_usleep(unsigned long useconds)
{
    if (useconds >= 1000000) {
        errno = EINVAL;
        return -1;
    } else {
        Sleep(useconds / 1000);
        return 0;
    }
}

#endif

#ifdef __cplusplus
}
#endif
