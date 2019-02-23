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
 * Minimal API for portably accessing files/assets at runtime, mainly
 * based on the Android Asset Management API.
 */

#pragma once

#include <sys/types.h> // off_t
#include <glimpse_log.h>

/* Note: currently compatible with Android's Asset mode enum which is
 * relied on internally
 */
enum {
    //GM_ASSET_MODE_UNKNOWN = 0,
    //GM_ASSET_MODE_RANDOM = 1,
    //GM_ASSET_MODE_STREAMING = 2,
    GM_ASSET_MODE_BUFFER = 3
};

struct gm_asset;

#ifdef __cplusplus
extern "C" {
#endif

/* To avoid assuming the path is specified by the GLIMPSE_ASSETS_ROOT
 * environment variable then middleware should set the path with
 * this api before opening assets
 */
void
gm_set_assets_root(struct gm_logger *log, const char *root);

const char *
gm_get_assets_root(void);

#ifdef USE_ANDROID_ASSET_MANAGER_API
#include <android/asset_manager.h>
void
gm_android_set_asset_manager(AAssetManager *manager);
#endif

struct gm_asset *
gm_asset_open(struct gm_logger *log,
              const char *path, int mode, char **err);

const void *
gm_asset_get_buffer(struct gm_asset *asset);

off_t
gm_asset_get_length(struct gm_asset *asset);

void
gm_asset_close(struct gm_asset *asset);

#ifdef __cplusplus
}
#endif
