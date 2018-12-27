/*
 * Copyright (C) 2018 Robert Bragg <robert@sixbynine.org>
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

#include "basename-compat.h"

#ifdef _WIN32

#include <stdlib.h>

char *
basename(char *filename)
{
    int len = strlen(filename);

    char *fname = alloca(len + 1);
    char *ext = alloca(len + 1);

    _splitpath_s(filename,
                 NULL, /* drive */
                 0,
                 NULL, /* dir */
                 0,
                 fname, /* base name (no ext) */
                 len + 1,
                 ext,
                 len + 1);
    snprintf(filename, len + 1, "%s%s", fname, ext);

    return filename;
}

char *
dirname(char *filename)
{
    int len = strlen(filename);

    char *drive = alloca(len + 1);
    char *dir = alloca(len + 1);

    _splitpath_s(filename,
                 drive, /* drive */
                 len + 1,
                 dir, /* dir */
                 len + 1,
                 NULL, /* base name (no ext) */
                 0,
                 NULL,
                 0);
    snprintf(filename, len + 1, "%s/%s", drive, dir);

    return filename;
}

#endif
