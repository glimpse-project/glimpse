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


/*
 * Provides a shared interface for logging but lets the application/middle-
 * ware handle the final IO.
 */

#pragma once

#include <stdlib.h>
#include <stdarg.h>

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

/* A given backtrace include instruction pointers for each frame and can
 * be resolved to symbols with code like this:
 *  FIXME
 */
struct gm_backtrace {
    int n_frames;
    const void **frame_pointers;
};

struct gm_logger *
gm_logger_new(void (*log_cb)(struct gm_logger *logger,
                             enum gm_log_level level,
                             const char *context,
                             struct gm_backtrace *backtrace,
                             const char *message,
                             va_list ap,
                             void *user_data),
              void *user_data);

/* XXX: This api is not thread-safe so it's assumed that the backtrace level
 * and size are set before the logger is used
 */
void
gm_logger_set_backtrace_level(struct gm_logger *logger,
                              int level);

/* XXX: This api is not thread-safe so it's assumed that the backtrace level
 * and size are set before the logger is used
 */
void
gm_logger_set_backtrace_size(struct gm_logger *logger,
                             int size);

/* At the point of logging then if we need to capture a backtrace then we only
 * save the instruction pointers for each frame. Later if the backtrace needs
 * to be displayed then those instruction pointers can be converted to human
 * readable strings...
 *
 * With libunwind we will query the function name and a byte offset formatted
 * like:
 *
 *   "gm_context_new()+0x80"
 *
 * It's then necessary to use a dissaembler to see what line the offset
 * really corresponds too.
 *
 * @frame_strings is allocated by the caller to be
 * (backtrace->n_frames * @string_lengths) bytes
 */
void
gm_logger_get_backtrace_strings(struct gm_logger *logger,
                                struct gm_backtrace *backtrace,
                                int string_lengths,
                                char *frame_strings);

void
gm_logger_set_abort_callback(struct gm_logger *logger,
                             void (*log_abort_cb)(struct gm_logger *logger,
                                                  void *user_data),
                             void *user_data);

void __attribute((noreturn))
gm_logger_abort(struct gm_logger *logger);

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
    if (!__builtin_expect((condition), 1)) { \
        gm_log(logger, GM_LOG_ASSERT, GM_LOG_CONTEXT, args); \
        gm_logger_abort(logger); \
    } \
} while(0)

#define gm_throw(logger, err, args...) do { \
    gm_assert(logger, err != NULL, args); \
    xasprintf(err, args); \
} while(0)

#ifdef __cplusplus
}
#endif
