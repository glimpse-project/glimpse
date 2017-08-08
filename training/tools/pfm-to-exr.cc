#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <dirent.h>
#include <stdint.h>
#include <libgen.h>

#include <ImfOutputFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>
#include <ImfChannelList.h>

#include <ImathBox.h>

#define DEBUG

#ifdef DEBUG
#define debug(ARGS...) printf(ARGS)
#else
#define debug(ARGS...) do {} while(0)
#endif

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static void *
xmalloc(size_t size)
{
    void *ret = malloc(size);
    if (ret == NULL)
        exit(1);
    return ret;
}

enum image_format {
    IMAGE_FORMAT_X8,
    IMAGE_FORMAT_XFLOAT,
};

struct image
{
    enum image_format format;
    int width;
    int height;
    int stride;

    union {
        uint8_t *data_u8;
        float *data_float;
    };
};


static struct image *
load_pfm(const char *filename)
{
    const char *asset_buf = NULL;
    long asset_buf_len;

    struct stat sb;

    int line_end[3] = { 0, 0, 0 };
    const char *type_line;
    const char *resolution_line;
    const char *scale_line;

    char *width_end = NULL;
    char *height_end = NULL;
    char *scale_end = NULL;

    int width, height;
    double scale;

    char *pfm_data_start;
    long pfm_data_len;

    struct image *ret = NULL;


    //return_if_fail(asset_manager != NULL, NULL);
    //return_if_fail(filename != NULL, NULL);

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %m\n", filename);
        return NULL;
    }

    fstat(fd, &sb);

    asset_buf = (const char *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (asset_buf) {
        asset_buf_len = sb.st_size;
        debug("Mapped PFM asset %p, len = %ld\n", asset_buf, asset_buf_len);
    } else {
        fprintf(stderr, "Failed to open shape predictor asset\n");
        return NULL;
    }

    int i, j = 0;
    for (i = 0; i < asset_buf_len && j < 3; i++) {
        if (asset_buf[i] == '\n')
            line_end[j++] = i;
    }
    if (i == asset_buf_len) {
        fprintf(stderr, "Parsing PFM header reached end of file\n");
        goto error_check_header;
    }

    type_line = (const char *)asset_buf;
    resolution_line = &asset_buf[line_end[0] + 1];
    scale_line = &asset_buf[line_end[1] + 1];

    if (strncmp(type_line, "Pf", 2) != 0) {
        fprintf(stderr, "Only support loading single component (Pf) portable float map files\n");
        goto error_check_header;
    }

    width = (int)strtoul(resolution_line, &width_end, 10);
    if (*width_end != ' ') {
        fprintf(stderr, "Expected <space> after PFM resoltion width\n");
        goto error_check_header;
    }

    height = (int)strtoul(width_end  + 1, &height_end, 10);
    if (height_end != (asset_buf + line_end[1])) {
        fprintf(stderr, "Nothing expected after PFM resoltion height\n");
        goto error_check_header;
    }

    if (width == 0 || width > 8192 || height == 0 || height > 8192) {
        fprintf(stderr, "Invalid PFM image width / height (%dx%d)\n", width, height);
        goto error_check_header;
    }

    scale = strtod(scale_line, &scale_end);
    if (scale_end != (asset_buf + line_end[2])) {
        fprintf(stderr, "Nothing expected after PFM scale");
        goto error_check_header;
    }
    //if (scale >= 0) {
    //    fprintf(stderr, "Only support loading little endian PFM files\n");
    //    goto error_check_header;
    // }

    pfm_data_start = scale_end + 1;
    pfm_data_len = sizeof(float) * width * height;

    ret = (struct image *)xmalloc(sizeof(struct image) + pfm_data_len);
    ret->format = IMAGE_FORMAT_XFLOAT;
    ret->stride = sizeof(float) * width;
    ret->width = width;
    ret->height = height;
    ret->data_float = (float *)(ret + 1);

    memcpy(ret->data_float, pfm_data_start, pfm_data_len);

    debug("read PFM %s (%dx%d) OK\n", filename, width, height);

error_check_header:
    munmap((void *)asset_buf, asset_buf_len);
error_mmap:
    close(fd);

    return ret;
}


static void
write_exr(struct image *image, const char *filename)
{
    Header header(image->width, image->height);
    header.channels().insert ("Y", Channel(FLOAT));

    OutputFile out_file(filename, header);

    FrameBuffer outFrameBuffer;
    outFrameBuffer.insert ("Y",
                           Slice (FLOAT,
                                  (char *)image->data_float,
                                  sizeof(float), // x stride,
                                  sizeof(float) * image->width)); // y stride

    out_file.setFrameBuffer(outFrameBuffer);
    out_file.writePixels(image->height);

    debug("wrote EXR %s (%dx%d) OK\n", filename, image->width, image->height);
}


int
main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s IN_PFM_FILE OUT_EXR_FILE\n", argv[0]);
        exit(1);
    }

    struct image *img = load_pfm(argv[1]);

    write_exr(img, argv[2]);

    return 0;
}
