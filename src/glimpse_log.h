
/*
 * Provides a shared interface for logging but lets the application/middle-
 * ware handle the final IO.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct gm_logger;

enum gm_log_level {
    GM_LOG_DEBUG,
    GM_LOG_INFO,
    GM_LOG_WARN,
    GM_LOG_ERROR,
    GM_LOG_ASSERT,
};

struct gm_logger *
gm_logger_new(void (*log_cb)(struct gm_logger *logger,
                             enum gm_log_level level,
                             const char *context,
                             const char *backtrace,
                             const char *message,
                             va_list ap,
                             void *user_data),
              void *user_data);

void
gm_logger_destroy(struct gm_logger *logger);

void
gm_logv(struct gm_logger *logger,
        enum gm_log_level level,
        const char *context,
        const char *format,
        va_list ap);

void
gm_log(struct gm_logger *logger,
       enum gm_log_level level,
       const char *context,
       const char *format,
       ...)
    __attribute__((__format__(printf, 4, 5)));

/* Compilation units should #undef this and re-define to something
 * more specific:
 */
#define GM_LOG_CONTEXT "Glimpse"

#define gm_debug(logger, args...) do { \
    gm_log(logger, GM_LOG_DEBUG, GM_LOG_CONTEXT, args); \
} while(0)

#define gm_info(logger, args...) do { \
    gm_log(logger, GM_LOG_INFO, GM_LOG_CONTEXT, args); \
} while(0)

#define gm_warn(logger, args...) do { \
    gm_log(logger, GM_LOG_WARN, GM_LOG_CONTEXT, args); \
} while(0)

#define gm_error(logger, args...) do { \
    gm_log(logger, GM_LOG_ERROR, GM_LOG_CONTEXT, args); \
} while(0)

#define gm_assert(logger, condition, args...) do { \
    if (__builtin_expect((condition), 1)) { \
        gm_log(logger, GM_LOG_ASSERT, GM_LOG_CONTEXT, args); \
        abort(); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif
