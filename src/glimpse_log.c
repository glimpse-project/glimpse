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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <limits.h>

#ifdef USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#include "xalloc.h"

#include "glimpse_log.h"

struct gm_logger {
    pthread_mutex_t lock;
    void (*callback)(struct gm_logger *logger,
                     enum gm_log_level level,
                     const char *context,
                     struct gm_backtrace *backtrace,
                     const char *message,
                     va_list ap,
                     void *user_data);
    void *callback_data;

    void (*abort_cb)(struct gm_logger *logger,
                     void *user_data);
    void *abort_cb_data;

    int backtrace_level;
    int backtrace_size;
};

struct gm_logger *
gm_logger_new(void (*log_cb)(struct gm_logger *logger,
                             enum gm_log_level level,
                             const char *context,
                             struct gm_backtrace *backtrace,
                             const char *format,
                             va_list ap,
                             void *user_data),
              void *user_data)
{
    struct gm_logger *logger = (struct gm_logger *)xcalloc(sizeof(*logger), 1);

    pthread_mutex_init(&logger->lock, NULL);
    logger->callback = log_cb;
    logger->callback_data = user_data;

    logger->backtrace_level = GM_LOG_ERROR;
    logger->backtrace_size = 10;

    return logger;
}

/* log messges with a level >= than this threshold will include a backtrace */
void
gm_logger_set_backtrace_level(struct gm_logger *logger,
                              int level)
{
    logger->backtrace_level = level;
}

void
gm_logger_set_backtrace_size(struct gm_logger *logger,
                             int size)
{
    logger->backtrace_size = size;
}

void
gm_logger_set_abort_callback(struct gm_logger *logger,
                             void (*log_abort_cb)(struct gm_logger *logger,
                                                  void *user_data),
                             void *user_data)
{
    logger->abort_cb = log_abort_cb;
    logger->abort_cb_data = user_data;
}

void __attribute((noreturn))
gm_logger_abort(struct gm_logger *logger)
{
    if (logger->abort_cb)
        logger->abort_cb(logger, logger->abort_cb_data);

    abort();
}

void
gm_logger_destroy(struct gm_logger *logger)
{
    xfree(logger);
}

#ifdef USE_LIBUNWIND
static int
get_backtrace(void **buffer, int skip, int size)
{
    unw_cursor_t cursor; unw_context_t uc;
    int i;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    for (i = 0; i < skip; i++) {
        if (unw_step(&cursor) <= 0)
            return 0;
    }

    for (i = 0; i < size && unw_step(&cursor) > 0; i++) {
        unw_word_t ip;
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        buffer[i] = (void *)(intptr_t)ip;
    }

    return i;
}
#endif

int
gm_backtrace(void **frame_pointers,
             int skip_n_frames,
             int n_frame_pointers)
{
#ifdef USE_LIBUNWIND
    return get_backtrace(frame_pointers,
                         skip_n_frames,
                         n_frame_pointers);
#else
    return 0;
#endif
}

void
gm_logger_get_backtrace_strings(struct gm_logger *logger,
                                struct gm_backtrace *backtrace,
                                int string_lengths,
                                char *frame_strings)
{
#ifdef USE_LIBUNWIND
    unw_cursor_t cursor; unw_context_t uc;
    char proc_name[string_lengths];
    unw_word_t proc_off;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    for (int i = 0; i < backtrace->n_frames; i++) {
        unw_set_reg(&cursor, UNW_REG_IP, (intptr_t)backtrace->frame_pointers[i]);
        unw_get_proc_name(&cursor,
                          proc_name,
                          sizeof(proc_name),
                          &proc_off);
        snprintf(frame_strings + string_lengths * i,
                 string_lengths,
                 "%s()+0x%lx",
                 proc_name,
                 proc_off);
    }
#else
    /* Shouldn't have ever recieved a backtrace */
    gm_assert(logger, 0, "spurious backtrace");
#endif
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

#ifdef USE_LIBUNWIND
    if (level >= logger->backtrace_level) {
        struct gm_backtrace bt;
        void *frame_pointers[10];

        bt.n_frames  = get_backtrace(frame_pointers, 1, 10);
        bt.frame_pointers = (const void **)frame_pointers;

        pthread_mutex_lock(&logger->lock);
        logger->callback(logger, level, context,
                         &bt,
                         format, ap, logger->callback_data);
        pthread_mutex_unlock(&logger->lock);
    } else
#endif
    {
        pthread_mutex_lock(&logger->lock);
        logger->callback(logger, level, context,
                         NULL, // TODO: support (optional) backtraces
                         format, ap, logger->callback_data);
        pthread_mutex_unlock(&logger->lock);
    }
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
