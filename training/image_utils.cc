#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include <png.h>

#include "tinyexr.h"
#include "half.hpp"
#include "xalloc.h"

#include "image_utils.h"

#define ARRAY_LEN(ARRAY) (sizeof(ARRAY)/sizeof(ARRAY[0]))

using half_float::half;

static IUReturnCode
_iu_verify_size(int width, int height, IUImageSpec* spec)
{
  // Verify image spec
  if (spec->width > 0)
    {
      if (spec->width != width)
        {
          fprintf(stderr, "Width %d != %d\n", width, spec->width);
          return NON_CONFORMANT;
        }
    }
  else
    {
      spec->width = width;
    }

  if (spec->height > 0)
    {
      if (spec->height != height)
        {
          fprintf(stderr, "Height %d != %d\n", height, spec->height);
          return NON_CONFORMANT;
        }
    }
  else
    {
      spec->height = height;
    }

  return SUCCESS;
}

static IUReturnCode
_iu_read_png_pal(png_structp png_ptr, png_infop info_ptr, void** output,
                 int* output_size)
{
  // Read out palette, if applicable
  if (!output ||
      png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_PALETTE)
    {
      return SUCCESS;
    }

  png_colorp palette;
  int palette_size;

  if (!png_get_PLTE(png_ptr, info_ptr, &palette, &palette_size))
    {
      fprintf(stderr, "Error reading palette\n");
      return PNG_ERR;
    }

  if (output_size)
    {
      *output_size = palette_size;
    }

  if (!(*output))
    {
      *output = xmalloc(palette_size * sizeof(png_color));
    }

  memcpy(*output, palette, palette_size * sizeof(png_color));

  return SUCCESS;
}

static bool
_check_spec_and_output(IUImageSpec *spec, uint8_t **output)
{
  if (*output)
    {
      if (!spec || spec->width == 0 || spec->height == 0 ||
          spec->format == IU_FORMAT_ANY)
        {
          fprintf(stderr, "Can't give output buffer without explicit spec\n");
          return false;
        }
    }

  return true;
}

IUReturnCode
iu_read_png_from_file(const char* filename, IUImageSpec* spec, uint8_t** output,
                      void** pal_output, int* pal_size)
{
  struct stat sb;

  if (stat(filename, &sb) < 0)
    {
      fprintf(stderr, "Failed to stat %s\n", filename);
      return IO_ERR;
    }

  FILE *fp = fopen(filename, "rb");
  if (!fp)
    {
      fprintf(stderr, "Failed to open %s\n", filename);
      return IO_ERR;
    }

  uint8_t *buf = (uint8_t *)xmalloc(sb.st_size);
  if (fread(buf, sb.st_size, 1, fp) != 1)
    {
      fprintf(stderr, "Failed to read %s\n", filename);
      return IO_ERR;
    }

  IUReturnCode ret = iu_read_png_from_memory(buf, sb.st_size, spec, output,
                                             pal_output, pal_size);

  fclose(fp);
  free(buf);

  return ret;
}

typedef struct {
  int len;
  int pos;
  uint8_t *buf;
} IUPngMemReadInfo;

static void
png_mem_read_cb(png_structp png_ptr, png_bytep data, png_size_t length)
{
  IUPngMemReadInfo *reader = (IUPngMemReadInfo *)png_get_io_ptr(png_ptr);
  int rem = reader->len - reader->pos;

  if (rem < (int)length)
    {
      png_error(png_ptr, "Ignoring request to read beyond end of PNG buffer");
      /* NB png_error() isn't expected to return*/
    }

  memcpy(data, reader->buf + reader->pos, length);
  reader->pos += length;
}

IUReturnCode
iu_read_png_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec,
                        uint8_t** output, void** pal_output, int* pal_size)
{
  int width, height;
  png_byte color_type;
  png_structp png_ptr;
  png_infop info_ptr;

  int row_stride;
  png_bytep *rows;

  IUPngMemReadInfo readinfo = { (int)len, 0, buffer };
  IUImageSpec default_spec = {0, 0, IU_FORMAT_U8};

  IUReturnCode ret = SUCCESS;

  if (!_check_spec_and_output(spec, output) != SUCCESS)
    return BAD_SPEC;

  if (!spec)
    {
      spec = &default_spec;
    }

  if (len <= 8)
    {
      fprintf(stderr, "Expected size of at least 8 bytes for PNG\n");
      return IO_ERR;
    }

  if (png_sig_cmp(buffer, 0, 8))
    {
      fprintf(stderr, "Error reading header\n");
      return IO_ERR;
    }

  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    {
      fprintf(stderr, "png_create_read_struct failed\n");
      return PNG_ERR;
    }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    {
      fprintf(stderr, "png_create_info_struct failed\n");
      ret = PNG_ERR;
      goto destroy_read_struct;
    }

  if (setjmp(png_jmpbuf(png_ptr)))
    {
      fprintf(stderr, "Failed to read PNG info\n");
      ret = PNG_ERR;
      goto destroy_info_struct;
    }

  png_set_read_fn(png_ptr, &readinfo, png_mem_read_cb);

  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);

  if (_iu_verify_size(width, height, spec) != SUCCESS)
    {
      ret = NON_CONFORMANT;
      goto destroy_info_struct;
    }

  color_type = png_get_color_type(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_GRAY &&
      color_type != PNG_COLOR_TYPE_PALETTE)
    {
      fprintf(stderr, "Expected an 8-bit color type\n");
      ret = NON_CONFORMANT;
      goto destroy_info_struct;
    }

  if (png_get_bit_depth(png_ptr, info_ptr) != 8)
    {
      fprintf(stderr, "Expected 8-bit pixel depth\n");
      ret = NON_CONFORMANT;
      goto destroy_info_struct;
    }

  if (!(*output))
    *output = (uint8_t *)xmalloc(spec->width * spec->height);

  if (setjmp(png_jmpbuf(png_ptr)))
    {
      fprintf(stderr, "PNG read failure\n");
      ret = IO_ERR;
      goto destroy_info_struct;
    }

  row_stride = png_get_rowbytes(png_ptr, info_ptr);
  if (row_stride != width)
    {
      fprintf(stderr, "Expected PNG row stride to match width\n");
      ret = BAD_FORMAT;
      goto destroy_info_struct;
    }

  rows = (png_bytep *)alloca(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++)
    rows[y] = (png_byte *)(*output + row_stride * y);

  png_read_image(png_ptr, rows);

  if (ret == SUCCESS && pal_output && pal_size)
    {
      ret = _iu_read_png_pal(png_ptr, info_ptr, pal_output, pal_size);
    }

destroy_info_struct:
  png_destroy_info_struct(png_ptr, &info_ptr);
destroy_read_struct:
  png_destroy_read_struct(&png_ptr, NULL, NULL);

  return ret;
}

IUReturnCode
iu_write_png_to_file(const char* filename, IUImageSpec* spec, void* data,
                     void* pal, int pal_size)
{
  png_structp png_ptr;
  png_infop info_ptr;
  png_byte color_type;
  png_bytep* rows;
  IUReturnCode ret = PNG_ERR;

  if (!spec)
    {
      fprintf(stderr, "Can't write image with no spec\n");
      return BAD_SPEC;
    }

  if (spec->format != IU_FORMAT_U8)
    {
      fprintf(stderr, "Only IU_FORMAT_U8 pngs are supported\n");
      return BAD_SPEC;
    }

  FILE *fp = fopen(filename, "wb");
  if (!fp)
    {
      fprintf(stderr, "Failed to open %s for writing\n", filename);
      return IO_ERR;
    }

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    {
      fprintf(stderr, "png_create_write_struct failed\n");
      goto iu_write_png_to_file_close;
    }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    {
      fprintf(stderr, "png_create_info_struct failed\n");
      goto iu_write_png_to_file_destroy_write_struct;
    }

  if (setjmp(png_jmpbuf(png_ptr)))
    {
      fprintf(stderr, "PNG write failure\n");
      goto iu_write_png_to_file_destroy_info_struct;
    }

  png_init_io(png_ptr, fp);

  color_type = pal ? PNG_COLOR_TYPE_PALETTE : PNG_COLOR_TYPE_GRAY;

  png_set_IHDR(png_ptr, info_ptr, spec->width, spec->height,
               8, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  if (pal)
    {
      png_set_PLTE(png_ptr, info_ptr, (png_color*)pal, pal_size);
    }

  png_write_info(png_ptr, info_ptr);

  rows = (png_bytep*)xmalloc(spec->height * sizeof(png_bytep));
  for (int y = 0; y < spec->height; y++)
    {
      rows[y] = (png_byte*)data + spec->width * y;
    }
  png_write_image(png_ptr, rows);
  xfree(rows);

  png_write_end(png_ptr, NULL);

  ret = SUCCESS;

iu_write_png_to_file_destroy_info_struct:
  png_destroy_info_struct(png_ptr, &info_ptr);
iu_write_png_to_file_destroy_write_struct:
  png_destroy_write_struct(&png_ptr, NULL);
iu_write_png_to_file_close:
  fclose(fp);

  return ret;
}

IUReturnCode
iu_read_exr_from_file(const char* filename, IUImageSpec* spec, void** output)
{
  struct stat sb;

  if (stat(filename, &sb) < 0)
    {
      fprintf(stderr, "Failed to stat %s\n", filename);
      return IO_ERR;
    }

  FILE *fp = fopen(filename, "rb");
  if (!fp)
    {
      fprintf(stderr, "Failed to open %s\n", filename);
      return IO_ERR;
    }

  uint8_t *buf = (uint8_t *)xmalloc(sb.st_size);
  if (fread(buf, sb.st_size, 1, fp) != 1)
    {
      fprintf(stderr, "Failed to read %s\n", filename);
      return IO_ERR;
    }

  IUReturnCode ret = iu_read_exr_from_memory(buf, sb.st_size, spec, output);

  fclose(fp);
  free(buf);

  return ret;
}

IUReturnCode
iu_read_exr_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec,
                        void** output)
{
  const unsigned char* memory = (const unsigned char*)buffer;
  EXRHeader header;

  const char *err = NULL;
  IUImageSpec blank_spec = {0, 0, IU_FORMAT_ANY};

  if (!spec)
    spec = &blank_spec;

  if (!_check_spec_and_output(spec, (uint8_t **)output) != SUCCESS)
    return BAD_SPEC;

  if (spec->format != IU_FORMAT_HALF &&
      spec->format != IU_FORMAT_FLOAT &&
      spec->format != IU_FORMAT_ANY)
    {
      fprintf(stderr, "Only support IU_FORMAT_HALF or IU_FORMAT_FLOAT spec format for EXR images");
      return BAD_SPEC;
    }

  EXRVersion version;
  int ret = ParseEXRVersionFromMemory(&version, memory, len);
  if (ret != TINYEXR_SUCCESS)
    {
      fprintf(stderr, "Error %02d reading EXR version\n", ret);
      return EXR_ERR;
    }

  if (version.multipart || version.non_image)
    {
      fprintf(stderr, "Can't load multipart or DeepImage EXR image\n");
      return EXR_ERR;
    }

  ret = ParseEXRHeaderFromMemory(&header, &version, memory, len, &err);

  if (ret != TINYEXR_SUCCESS)
    {
      fprintf(stderr, "Error %02d reading EXR header: %s\n", ret, err);
      return EXR_ERR;
    }

  int width = header.data_window[2] - header.data_window[0] + 1;
  int height = header.data_window[3] - header.data_window[1] + 1;

  if (_iu_verify_size(width, height, spec) != SUCCESS)
    {
      FreeEXRHeader(&header);
      return NON_CONFORMANT;
    }

  int channel = -1;
  for (int i = 0; i < header.num_channels; i++)
    {
      const char *names[] = { "Y", "R", "G", "B" };
      for (unsigned j = 0; j < ARRAY_LEN(names); j++) {
          if (strcmp(names[j], header.channels[i].name) == 0) {
              channel = i;
              break;
          }
      }
      if (channel > 0)
          break;
    }

  if (channel == -1)
    {
      fprintf(stderr, "Failed to find R, G, B or Y channel in EXR file\n");
      FreeEXRHeader(&header);
      return NON_CONFORMANT;
    }

  int pixel_type = header.channels[channel].pixel_type;
  if (pixel_type != TINYEXR_PIXELTYPE_HALF && pixel_type != TINYEXR_PIXELTYPE_FLOAT)
    {
      fprintf(stderr, "Can only decode EXR images with FLOAT or HALF data\n");
      FreeEXRHeader(&header);
      return NON_CONFORMANT;
    }

  EXRImage exr_image;
  InitEXRImage(&exr_image);

  if (spec->format == IU_FORMAT_ANY)
    {
      spec->format = pixel_type == TINYEXR_PIXELTYPE_HALF ?
        IU_FORMAT_HALF : IU_FORMAT_FLOAT;
    }
  else
    {
      header.requested_pixel_types[channel] = spec->format == IU_FORMAT_HALF ?
        TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
    }

  if (LoadEXRImageFromMemory(&exr_image, &header, memory, len, &err) !=
      TINYEXR_SUCCESS)
    {
      fprintf(stderr, "TinyEXR failed to load EXR from memory: %s\n", err);
      FreeEXRHeader(&header);
      return BAD_FORMAT;
    }

  int bpp = spec->format == IU_FORMAT_HALF ? 2 : 4;

  if (!(*output))
    {
      *output = xmalloc(spec->width * spec->height * bpp);
    }

  memcpy(*output,
         &exr_image.images[channel][0],
         spec->width * spec->height * bpp);

  FreeEXRImage(&exr_image);
  FreeEXRHeader(&header);

  return SUCCESS;
}

IUReturnCode
iu_write_exr_to_file(const char* filename,
                     IUImageSpec* spec,
                     void* data,
                     IUImageFormat format)
{
  if (!spec)
    {
      fprintf(stderr, "Can't write image with no spec\n");
      return BAD_SPEC;
    }

  if (spec->format != IU_FORMAT_HALF && spec->format != IU_FORMAT_FLOAT)
    {
      fprintf(stderr, "Must specify IU_FORMAT_HALF or IU_FORMAT_FLOAT format in spec to write EXR\n");
      return BAD_SPEC;
    }

  if (spec->width == 0 || spec->height == 0)
    {
      fprintf(stderr, "Must specify width/height when writing EXR\n");
      return BAD_SPEC;
    }

  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage exr_image;
  InitEXRImage(&exr_image);

  exr_image.num_channels = 1;
  exr_image.width = spec->width;
  exr_image.height = spec->height;
  exr_image.images = (unsigned char**)(&data);

  header.num_channels = 1;
  EXRChannelInfo channel_info;
  header.channels = &channel_info;
  strcpy(channel_info.name, "Y");

  header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

  int input_format = (spec->format == IU_FORMAT_FLOAT ?
                      TINYEXR_PIXELTYPE_FLOAT : TINYEXR_PIXELTYPE_HALF);
  int final_format = (format == IU_FORMAT_FLOAT ?
                      TINYEXR_PIXELTYPE_FLOAT : TINYEXR_PIXELTYPE_HALF);
  header.pixel_types = &input_format;
  header.requested_pixel_types = &final_format;

  const char *err = NULL;
  if (SaveEXRImageToFile(&exr_image, &header, filename, &err) !=
      TINYEXR_SUCCESS)
    {
      fprintf(stderr, "Failed to save EXR: %s\n", err);
      return EXR_ERR;
    }

  return SUCCESS;
}

const char *
iu_code_to_string(IUReturnCode code)
{
  switch (code)
    {
    case SUCCESS:
      return "OK";
    case BAD_SPEC:
      return "Bad Specification";
    case NON_CONFORMANT:
      return "Non Conformant";
    case BAD_FORMAT:
      return "Bad Format";
    case IO_ERR:
      return "File IO error";
    case PNG_ERR:
      return "libpng error";
    case EXR_ERR:
      return "TinyEXR error";

    // default intentionally omitted so we get a compiler
    // warning for not handling new error codes
    }

    return "Unknown";
}
