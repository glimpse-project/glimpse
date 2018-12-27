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


#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#else
#define TARGET_OS_MAC 0
#define TARGET_OS_IOS 0
#define TARGET_OS_OSX 0
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define strdup(X) _strdup(X)
#define open _open
#define close _close
#define fdopen _fdopen
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "xalloc.h"
#include "glimpse_log.h"
#include "glimpse_assets.h"
#include "glimpse_os.h"


struct gm_asset
{
    /* Note: It's not assumed that we will be using the Asset Manager API on
     * Android because it's sometimes more convenient during development to
     * load assets from external storage.
     */
#ifdef USE_ANDROID_ASSET_MANAGER_API
    AAsset *native;
#else
    char *path;
    uint8_t *buf;
    bool buf_is_mmaped;
    int mode;
    int fd;
    off_t file_len;
    bool mapped;
#ifdef _WIN32
    HANDLE mapping;
#endif
#endif
};

static char *glimpse_assets_root;

void
gm_set_assets_root(struct gm_logger *log, const char *root)
{
    if (glimpse_assets_root)
        free(glimpse_assets_root);
    if (root && strlen(root) > 0)
        glimpse_assets_root = strdup(root);
    else
        glimpse_assets_root = strdup(".");
    gm_debug(log, "Set Assets Root to \"%s\"", glimpse_assets_root);
}

const char *
gm_get_assets_root(void)
{
    /* An empty assets root will break assumptions made while deriving the
     * path for glimpse assets and we don't want to let things inadvertantly
     * create paths relative to the root of the filesystem
     */
    assert(glimpse_assets_root && strlen(glimpse_assets_root) > 0);

    return glimpse_assets_root;
}

#ifdef USE_ANDROID_ASSET_MANAGER_API
static AAssetManager *asset_manager;

void
gm_android_set_asset_manager(AAssetManager *manager)
{
    asset_manager = manager;
}

struct gm_asset *
gm_asset_open(struct gm_logger *log,
              const char *path, int mode, char **err)
{
    gm_assert(log, asset_manager != NULL,
              "gm_android_set_asset_manager not called");

    AAsset *native = AAssetManager_open(asset_manager, path, mode);
    if (native) {
        struct gm_asset *ret = xmalloc(sizeof(*ret));
        ret->native = native;
        return ret;
    } else {
        gm_throw(log, err, "Failed to open %s\n", path);
        return NULL;
    }
}

const void *
gm_asset_get_buffer(struct gm_asset *asset)
{
    return AAsset_getBuffer(asset->native);
}

off_t
gm_asset_get_length(struct gm_asset *asset)
{
    return AAsset_getLength(asset->native);
}

void
gm_asset_close(struct gm_asset *asset)
{
    AAsset_close(asset->native);
}

#else

#define xsnprintf(dest, len, fmt, ...) do { \
        if (snprintf(dest, len, fmt,  __VA_ARGS__) >= (int)len) \
            exit(1); \
    } while(0)

struct gm_asset *
gm_asset_open(struct gm_logger *log,
              const char *path, int mode, char **err)
{
    int fd;
    struct stat sb;
    uint8_t *buf = NULL;
    bool buf_is_mmaped = false;
#ifdef _WIN32
    HANDLE mapping = NULL;
#endif
    struct gm_asset *asset;
    char *root = NULL;

    root = glimpse_assets_root;
    if (!root)
        root = "./";

    int max_len = strlen(path) + strlen(root) + 2;
    char *full_path = alloca(max_len);
    xsnprintf(full_path, max_len, "%s/%s", root, path);

    int oflags = O_RDONLY;
#ifdef _WIN32
    oflags |= O_BINARY;
#endif
#if defined(__unix__) || defined(__APPLE__)
    oflags |= O_CLOEXEC;
#endif
    fd = open(full_path, oflags);
    if (fd < 0) {
        gm_throw(log, err, "Failed to open %s: %s",
                 full_path, strerror(errno));
        return NULL;
    }

    if (fstat(fd, &sb) < 0) {
        gm_throw(log, err, "Failed to stat %s file descriptor: %s",
                 full_path, strerror(errno));
        return NULL;
    }

    switch (mode) {
    case GM_ASSET_MODE_BUFFER:
#ifdef __unix__
        buf = (uint8_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (buf)
            buf_is_mmaped = true;
#elif defined(_WIN32)
        mapping = CreateFileMapping(_get_osfhandle(fd),
                                    NULL, 
                                    PAGE_READONLY,
                                    0, // high 32 bits of max file size
                                    0, // low 32 bits of max file size (both 0 = full size)
                                    NULL); // un-named mapping
        if (mapping) {
            buf = MapViewOfFile(mapping,
                                FILE_MAP_READ,
                                0, // high 32 bits of file offset
                                0, // low 32 bits of file offset
                                0); // end of mapping (0 = end of file)
            if (buf)
                buf_is_mmaped = true;
            else {
                CloseHandle(mapping);
                mapping = NULL;
            }
        }
#endif
        if (!buf) {
            FILE *fp = fdopen(fd, "r");
            if (!fp) {
                close(fd);
                gm_throw(log, err, "Failed to open buffered IO stream");
                return NULL;
            }
            buf = xmalloc(sb.st_size);
            if (fread(buf, sb.st_size, 1, fp) != 1) {
                xfree(buf);
                close(fd);
                gm_throw(log, err, "Failed to read %d bytes from buffered IO stream %p for file %s",
                         (int)sb.st_size, fp, full_path);
                return NULL;
            }
        }
        break;
    }

    asset = (struct gm_asset *)xcalloc(sizeof(*asset), 1);
    asset->mode = mode;
    asset->fd = fd;
    asset->file_len = sb.st_size;
    asset->path = strdup(path);
    asset->buf = buf;
    asset->buf_is_mmaped = buf_is_mmaped;
#ifdef _WIN32
    asset->mapping = mapping;
#endif

    return asset;
}

const void *
gm_asset_get_buffer(struct gm_asset *asset)
{
    return (void *)asset->buf;
}

off_t
gm_asset_get_length(struct gm_asset *asset)
{
    return asset->file_len;
}

void
gm_asset_close(struct gm_asset *asset)
{
    switch (asset->mode) {
    case GM_ASSET_MODE_BUFFER:
        if (asset->buf) {
#if defined(__unix__) || defined(__APPLE__)
            if (asset->buf_is_mmaped) {
                munmap(asset->buf, asset->file_len);
                asset->buf = NULL;
            }
#elif defined(_WIN32)
            if (asset->buf_is_mmaped) {
                UnmapViewOfFile(asset->buf);
                CloseHandle(asset->mapping);
                asset->buf = NULL;
            }
#endif
            if (asset->buf) {
                xfree(asset->buf);
                asset->buf = NULL;
            }
        }
        break;
    }
    close(asset->fd);
    free(asset->path);
    free(asset);
}

#endif
