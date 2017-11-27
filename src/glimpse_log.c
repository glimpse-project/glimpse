#include <stdarg.h>
#include <string.h>
#include <pthread.h>

#include "xalloc.h"

#include "glimpse_log.h"

struct gm_logger {
    pthread_mutex_t lock;
    void (*callback)(struct gm_logger *logger,
                     enum gm_log_level level,
                     const char *context,
                     const char *backtrace,
                     const char *message,
                     va_list ap,
                     void *user_data);
    void *callback_data;
};

struct gm_logger *
gm_logger_new(void (*log_cb)(struct gm_logger *logger,
                             enum gm_log_level level,
                             const char *context,
                             const char *backtrace,
                             const char *format,
                             va_list ap,
                             void *user_data),
              void *user_data)
{
    struct gm_logger *logger = (struct gm_logger *)xcalloc(sizeof(*logger), 1);

    pthread_mutex_init(&logger->lock, NULL);
    logger->callback = log_cb;
    logger->callback_data = user_data;

    return logger;
}

void
gm_logger_destroy(struct gm_logger *logger)
{
    xfree(logger);
}

void
gm_logv(struct gm_logger *logger,
        enum gm_log_level level,
        const char *context,
        const char *format,
        va_list ap)
{
    /* For consistency we strip any newline from the end of the message */

    int fmt_len = strlen(format);
    char tmp[fmt_len];

    if (fmt_len && format[fmt_len - 1] == '\n') {
        memcpy(tmp, format, fmt_len - 1);
        tmp[fmt_len - 1] = '\0';
        format = (const char *)tmp;
    }

    pthread_mutex_lock(&logger->lock);
    logger->callback(logger, level, context,
                     NULL, // TODO: support (optional) backtraces
                     format, ap, logger->callback_data);
    pthread_mutex_unlock(&logger->lock);
}

void
gm_log(struct gm_logger *logger,
       enum gm_log_level level,
       const char *context,
       const char *format,
       ...)
{
    va_list ap;

    va_start(ap, format);
    gm_logv(logger, level, context, format, ap);
    va_end(ap);
}

