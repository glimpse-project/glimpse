
/*
 * Minimal API for portably accessing files/assets at runtime, mainly
 * based on the Android Asset Management API.
 */

#pragma once

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

#ifdef ANDROID
#include <android/asset_manager.h>
void
gm_android_set_asset_manager(AAssetManager *manager);
#endif

struct gm_asset *
gm_asset_open(const char *path, int mode, char **err);

const void *
gm_asset_get_buffer(struct gm_asset *asset);

off_t
gm_asset_get_length(struct gm_asset *asset);

void
gm_asset_close(struct gm_asset *asset);

#ifdef __cplusplus
}
#endif
