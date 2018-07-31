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


#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "xalloc.h"
#include "glimpse_log.h"
#include "glimpse_assets.h"


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
    int mode;
    int fd;
    off_t file_len;
    bool mapped;
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
    struct gm_asset *asset;
    char *root = NULL;

    root = glimpse_assets_root;
    if (!root)
        root = "./";

    int max_len = strlen(path) + strlen(root) + 2;
    char *full_path = alloca(max_len);
    xsnprintf(full_path, max_len, "%s/%s", root, path);

    fd = open(full_path, O_RDONLY|O_CLOEXEC);
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
        buf = (uint8_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (!buf) {
            close(fd);
            gm_throw(log, err, "Failed to mmap %s: %s",
                     full_path, strerror(errno));
            return NULL;
        }
        break;
    }

    asset = (struct gm_asset *)xcalloc(sizeof(*asset), 1);
    asset->mode = mode;
    asset->fd = fd;
    asset->file_len = sb.st_size;
    asset->path = strdup(path);
    asset->buf = buf;

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
        if (asset->buf)
            munmap(asset->buf, asset->file_len);
        break;
    }
    close(asset->fd);
    free(asset->path);
    free(asset);
}

#endif
