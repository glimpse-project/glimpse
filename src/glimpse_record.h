/*
 * Copyright (C) 2018 Glimp IP Ltd
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

#pragma once

#include <list>
#include <stdbool.h>
#include "glimpse_log.h"
#include "glimpse_context.h"
#include "glimpse_device.h"

struct gm_recording;

#ifndef __cplusplus
extern "C" {
#endif

struct gm_recording *
gm_recording_init(struct gm_logger *log,
                  struct gm_device *device,
                  const char *recordings_path,
                  const char *rel_path,
                  bool overwrite,
                  uint64_t max_io_buffer_size,
                  char **err);

void gm_recording_append_frame(struct gm_recording *recording,
                               struct gm_frame *frame);

void
gm_recording_stop(struct gm_recording *recording);

bool
gm_recording_is_stopped(struct gm_recording *recording);

bool
gm_recording_is_async_io_finished(struct gm_recording *recording);

uint64_t
gm_recording_get_io_buffer_size(struct gm_recording *recording);

uint64_t
gm_recording_get_max_io_buffer_size(struct gm_recording *recording);

bool
gm_recording_close(struct gm_recording *recording);


#ifndef __cplusplus
}
#endif
