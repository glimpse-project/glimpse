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


#ifndef __UTILS__
#define __UTILS__

#include <stdint.h>
#include <sys/types.h>
#include <math.h>

#include "half.hpp"

#define JIP_VERSION 0

#define vector(type,size) type __attribute__ ((vector_size(sizeof(type)*(size))))

typedef vector(int, 2) Int2D;
typedef vector(float, 4) UVPair;

typedef struct {
    Int2D xy;
    int i;
} Int3D;

template<typename FloatT>
inline float
sample_uv(FloatT* depth_image, int width, int height,
          Int2D pixel, float depth, UVPair uv)
{
#if 0
    // This code path is slower. gcc is cleverer than me, leaving this here as
    // a reminder.
    vector(float, 4) uv_pixel = { (float)pixel[0], (float)pixel[1],
                                  (float)pixel[0], (float)pixel[1] };
    uv_pixel += uv / depth;

    vector(float, 4) extents = { (float)width, (float)height,
                                 (float)width, (float)height };
    vector(int, 4) mask = (uv_pixel >= 0.f && uv_pixel < extents);

    float upixel = (mask[0] && mask[1]) ?
        depth_image[(((uint32_t)uv_pixel[1] * width) + (uint32_t)uv_pixel[0])] : 1000.f;
    float vpixel = (mask[2] && mask[3]) ?
        depth_image[(((uint32_t)uv_pixel[3] * width) + (uint32_t)uv_pixel[2])] : 1000.f;

    return upixel - vpixel;
#else
    Int2D u = { (int)(pixel[0] + uv[0] / depth),
                (int)(pixel[1] + uv[1] / depth) };
    Int2D v = { (int)(pixel[0] + uv[2] / depth),
                (int)(pixel[1] + uv[3] / depth) };

    float upixel = (u[0] >= 0 && u[0] < (int)width &&
                    u[1] >= 0 && u[1] < (int)height) ?
        (float)depth_image[((u[1] * width) + u[0])] : 1000.f;
    float vpixel = (v[0] >= 0 && v[0] < (int)width &&
                    v[1] >= 0 && v[1] < (int)height) ?
        (float)depth_image[((v[1] * width) + v[0])] : 1000.f;

    return upixel - vpixel;
#endif
}

typedef struct {
    int32_t hours;
    int32_t minutes;
    int32_t seconds;
} TimeForDisplay;

inline TimeForDisplay
get_time_for_display(struct timespec* begin, struct timespec* end)
{
    uint32_t elapsed;
    TimeForDisplay display;

    elapsed = (end->tv_sec - begin->tv_sec);
    elapsed += (end->tv_nsec - begin->tv_nsec) / 1000000000;

    display.seconds = elapsed % 60;
    display.minutes = elapsed / 60;
    display.hours = display.minutes / 60;
    display.minutes = display.minutes % 60;

    return display;
}

#endif /* __UTILS__ */

