/*
 * Copyright (C) 2014 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdbool.h>

#include "glimpse_mutex.h"

#ifndef _WIN32

#include <errno.h>
#include <pthread.h>

void
gm_mutex_init(gm_mutex_t *mutex)
{
    pthread_mutex_t *m = (pthread_mutex_t *)mutex;
    pthread_mutexattr_t attr;

    /* NB: Windows CriticalSection and Mutex APIs don't support non-recursive
     * semantics, so for consistency...
     */
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

    pthread_mutex_init(m, NULL);
}

void
gm_mutex_destroy(gm_mutex_t *mutex)
{
    pthread_mutex_t *m = (pthread_mutex_t *)mutex;
    pthread_mutex_destroy(m);
}

void
gm_mutex_lock(gm_mutex_t *mutex)
{
    pthread_mutex_t *m = (pthread_mutex_t *)mutex;
    pthread_mutex_lock(m);
}

void
gm_mutex_unlock(gm_mutex_t *mutex)
{
    pthread_mutex_t *m = (pthread_mutex_t *)mutex;
    pthread_mutex_unlock(m);
}

bool
gm_mutex_trylock(gm_mutex_t *mutex)
{
    pthread_mutex_t *m = (pthread_mutex_t *)mutex;
    int ret = pthread_mutex_trylock(m);

    if (ret == 0)
        return true;
    else if (ret == EBUSY)
        return false;
    else {
        return false;
    }
}

#else

void
gm_mutex_init(gm_mutex_t *mutex)
{
    InitializeCriticalSectionAndSpinCount((CRITICAL_SECTION *)mutex,
                                          2000);
}

void
gm_mutex_destroy(gm_mutex_t *mutex)
{
    DeleteCriticalSection((CRITICAL_SECTION *)mutex);
}

void
gm_mutex_lock(gm_mutex_t *mutex)
{
    EnterCriticalSection((CRITICAL_SECTION *)mutex);
}

void
gm_mutex_unlock(gm_mutex_t *mutex)
{
    LeaveCriticalSection((CRITICAL_SECTION *)mutex);
}

bool
gm_mutex_trylock(gm_mutex_t *mutex)
{
    return TryEnterCriticalSection((CRITICAL_SECTION *)mutex);
}

#endif
