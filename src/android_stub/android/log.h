#pragma once

#include <stdio.h>
#include <stdarg.h>

enum {
  ANDROID_LOG_UNKNOWN = 0,
  ANDROID_LOG_DEFAULT,
  ANDROID_LOG_VERBOSE,
  ANDROID_LOG_DEBUG,
  ANDROID_LOG_INFO,
  ANDROID_LOG_WARN,
  ANDROID_LOG_ERROR,
  ANDROID_LOG_FATAL,
  ANDROID_LOG_SILENT
};

#ifdef __cplusplus
extern "C" {
#endif

/* XXX: we should probably put this into an object file with a lock to properly
 * serialize log messages from multiple threads
 */
static int
__android_log_print(int prior, const char *tag, const char *fmt, ...)
    __attribute__((unused))
    __attribute__((__format__(printf, 3, 4)));

static int
__android_log_print(int prior, const char *tag, const char *fmt, ...)
{
    int ret;
    va_list ap;

    va_start(ap, fmt);

    ret = fprintf(stderr, "%s: ", tag);
    ret += vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");

    va_end(ap);

    return ret;
}

static void
__android_log_assert(const char *cond,
                     const char *tag,
                     const char *fmt,...)
    __attribute__((unused))
    __attribute__((__noreturn__))
    __attribute__((__format__(printf, 3, 4)));

static void
__android_log_assert(const char *cond,
                     const char *tag,
                     const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    fprintf(stderr, "%s: assertion %s failed:", tag, cond);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);

    exit(1);
}

#ifdef __cplusplus
}
#endif
