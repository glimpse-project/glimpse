#pragma once


#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include <xalloc.h>

enum {
    AASSET_MODE_UNKNOWN = 0,
    AASSET_MODE_RANDOM = 1,
    AASSET_MODE_STREAMING = 2,
    AASSET_MODE_BUFFER = 3
};

typedef struct {
    char *filename;
    uint8_t *buf;
    int mode;
    int fd;
    off_t file_len;
    bool mapped;
} AAsset;

typedef struct _AAssetManager AAssetManager;


#ifdef __cplusplus
extern "C" {
#endif

static AAsset *
AAssetManager_open(AAssetManager *mgr, const char *filename, int mode)
    __attribute__((unused));

static AAsset *
AAssetManager_open(AAssetManager *mgr, const char *filename, int mode)
{
    int fd;
    struct stat sb;
    AAsset *asset;
    
    fd = open(filename, O_RDWR|O_CLOEXEC);
    if (fd < 0)
        return NULL;

    if (fstat(fd, &sb) < 0)
        return NULL;

    asset = (AAsset *)xcalloc(sizeof(AAsset), 1);
    asset->mode = mode;
    asset->fd = fd;
    asset->file_len = sb.st_size;
    asset->filename = strdup(filename);

    switch (mode) {
    case AASSET_MODE_BUFFER:
        asset->buf = (uint8_t *)mmap(NULL, asset->file_len,
                                     PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        break;
    }

    return asset;
}

static const void *
AAsset_getBuffer(AAsset *asset)
    __attribute__((unused));

static const void *
AAsset_getBuffer(AAsset *asset)
{
    return (void *)asset->buf;
}

static const off_t
AAsset_getLength(AAsset *asset)
    __attribute__((unused));

static const off_t
AAsset_getLength(AAsset *asset)
{
    return asset->file_len;
}

static void
AAsset_close(AAsset *asset)
    __attribute__((unused));

static void
AAsset_close(AAsset *asset)
{
    switch (asset->mode) {
    case AASSET_MODE_BUFFER:
        if (asset->buf)
            munmap(asset->buf, asset->file_len);
        break;
    }
    close(asset->fd);
    free(asset->filename);
    free(asset);
}


#ifdef __cplusplus
}
#endif

