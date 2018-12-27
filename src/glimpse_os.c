
#include <stdbool.h>
#include <stdint.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#elif defined(_WIN32)
#include <windows.h>
#include <direct.h>
#endif

#include "glimpse_os.h"
#include "glimpse_log.h"

uint64_t
gm_os_get_time(void)
{
#if defined(__unix__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
#elif defined(_WIN32)
    static double qpc_scale = 0;
    LARGE_INTEGER li;

    if (!qpc_scale) {
        QueryPerformanceFrequency(&li);
        qpc_scale = 1e9 / li.QuadPart;
    }
    QueryPerformanceCounter(&li);
    return li.QuadPart * qpc_scale;
#endif
}

bool
gm_os_mkdir(struct gm_logger *log, const char *path, char **err)
{
#if defined(__unix__) || defined(__APPLE__)
    int status = mkdir(path, 0777);
    if (status < 0 && errno != EEXIST) {
        gm_throw(log, err, "Failed to make directory %s: %s",
                 path, strerror(errno));
        return false;
    }
    return true;
#elif defined(_WIN32)
    int status = _mkdir(path);
    if (status < 0 && errno != EEXIST) {
        gm_throw(log, err, "Failed to make directory %s: %s",
                 path, strerror(errno));
        return false;
    }
    return true;
#endif
}
