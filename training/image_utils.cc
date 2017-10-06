
#include "image_utils.h"

#include <string.h>
#include <png.h>

#include "tinyexr.h"
#include "half.hpp"
#include "xalloc.h"

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
_iu_verify_png(png_structp png_ptr, png_infop info_ptr, IUImageSpec* spec)
{
  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);

  if (_iu_verify_size(width, height, spec) != SUCCESS)
    {
      return NON_CONFORMANT;
    }

  // Verify this is an 8-bit png we're reading
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_GRAY &&
      color_type != PNG_COLOR_TYPE_PALETTE)
    {
      fprintf(stderr, "Expected an 8-bit color type\n");
      return NON_CONFORMANT;
    }

  if (png_get_bit_depth(png_ptr, info_ptr) != 8)
    {
      fprintf(stderr, "Expected 8-bit pixel depth\n");
      return NON_CONFORMANT;
    }

  spec->depth = 8;

  return SUCCESS;
}

static IUReturnCode
_iu_close_png_from_file(FILE* fp, png_structp png_ptr, png_infop info_ptr)
{
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  if (fclose(fp) != 0)
    {
      fprintf(stderr, "Error closing png file\n");
      return IO_ERR;
    }
  return SUCCESS;
}

static IUReturnCode
_iu_open_png_from_file(const char* filename, FILE** fp, png_structp* png_ptr,
                       png_infop* info_ptr)
{
  unsigned char header[8]; // 8 is the maximum size that can be checked
  IUReturnCode ret = SUCCESS;

  // Open png file
  if (!(*fp = fopen(filename, "rb")))
    {
      fprintf(stderr, "Failed to open image\n");
      ret = IO_ERR;
      goto open_png_from_file_return;
    }

  // Read header from png file
  if (fread(header, 1, 8, *fp) != 8)
    {
      fprintf(stderr, "Error reading header\n");
      ret = IO_ERR;
      goto open_png_from_file_close_file;
    }

  if (png_sig_cmp(header, 0, 8))
    {
      fprintf(stderr, "Expected PNG file\n");
      ret = BAD_FORMAT;
      goto open_png_from_file_close_file;
    }

  // Create structures for reading png data
  *png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!(*png_ptr))
    {
      fprintf(stderr, "png_create_read_struct failed\n");
      ret = PNG_ERR;
      goto open_png_from_file_close_file;
    }

  *info_ptr = png_create_info_struct(*png_ptr);
  if (!(*info_ptr))
    {
      fprintf(stderr, "png_create_info_struct failed\n");
      ret = PNG_ERR;
      goto open_png_from_file_destroy_read_struct;
    }

  // Set error handler
  if (setjmp(png_jmpbuf(*png_ptr)))
    {
      fprintf(stderr, "libpng setjmp failure\n");
      return PNG_ERR;
    }

  png_init_io(*png_ptr, *fp);
  png_set_sig_bytes(*png_ptr, 8);

  png_read_info(*png_ptr, *info_ptr);

  return ret;

  png_destroy_info_struct(*png_ptr, info_ptr);
open_png_from_file_destroy_read_struct:
  png_destroy_read_struct(png_ptr, NULL, NULL);
open_png_from_file_close_file:
  if (fclose(*fp) != 0)
    {
      fprintf(stderr, "Error closing png file\n");
      if (ret == SUCCESS)
        {
          ret = IO_ERR;
        }
    }
open_png_from_file_return:
  return ret;
}

static IUReturnCode
_iu_read_png(png_structp* png_ptr, png_infop* info_ptr, IUImageSpec* spec,
             void** output)
{
  // Allocate output if necessary
  if (!(*output))
    {
      *output = xmalloc(spec->width * spec->height);
    }

  // Set error handler
  if (setjmp(png_jmpbuf(*png_ptr)))
    {
      fprintf(stderr, "libpng setjmp failure\n");
      return PNG_ERR;
    }

  // Start reading data
  int row_stride = png_get_rowbytes(*png_ptr, *info_ptr);
  png_bytep* input_rows = (png_bytep *)
    xmalloc(sizeof(png_bytep*) * spec->height);
  png_bytep input_data = (png_bytep)
    xmalloc(row_stride * spec->height * sizeof(png_bytep));

  for (int y = 0; y < spec->height; y++)
    {
      input_rows[y] = (png_byte*)input_data + row_stride * y;
    }

  png_read_image(*png_ptr, input_rows);

  // Copy label image data into training context struct
  for (int y = 0, src_idx = 0, dst_idx = 0;
       y < spec->height; y++, src_idx += row_stride, dst_idx += spec->width)
    {
      memcpy(&(((char *)*output)[dst_idx]), &input_data[src_idx], spec->width);
    }

  // Free data associated with PNG reading
  xfree(input_rows);
  xfree(input_data);

  return SUCCESS;
}

IUReturnCode
iu_read_png_from_file(const char* filename, IUImageSpec* spec, void** output)
{
  FILE* fp;
  png_structp png_ptr;
  png_infop info_ptr;

  IUImageSpec blank_spec = {0, 0, 8, 1};
  IUReturnCode ret = SUCCESS;

  if (!spec)
    {
      spec = &blank_spec;
    }

  if (spec->depth != 0 && spec->depth != 8)
    {
      fprintf(stderr, "Spec requires non-8-bit pixel depth\n");
      return BAD_SPEC;
    }

  // Open png file
  ret = _iu_open_png_from_file(filename, &fp, &png_ptr, &info_ptr);
  if (ret != SUCCESS)
    {
      return ret;
    }

  // Verify png file
  ret = _iu_verify_png(png_ptr, info_ptr, spec);
  if (ret != SUCCESS)
    {
      _iu_close_png_from_file(fp, png_ptr, info_ptr);
      return ret;
    }

  ret = _iu_read_png(&png_ptr, &info_ptr, spec, output);

  IUReturnCode close_ret = _iu_close_png_from_file(fp, png_ptr, info_ptr);

  return (ret == SUCCESS) ? close_ret : ret;
}

IUReturnCode
iu_verify_png_from_file(const char* filename,
                        IUImageSpec* spec)
{
  FILE* fp;
  png_structp png_ptr;
  png_infop info_ptr;

  IUImageSpec blank_spec = {0, 0, 8, 1};
  IUReturnCode ret = SUCCESS;

  if (!spec)
    {
      spec = &blank_spec;
    }

  if (spec->depth != 0 && spec->depth != 8)
    {
      fprintf(stderr, "Spec requires non-8-bit pixel depth\n");
      return BAD_SPEC;
    }


  // Open png file
  ret = _iu_open_png_from_file(filename, &fp, &png_ptr, &info_ptr);
  if (ret != SUCCESS)
    {
      return ret;
    }

  if (setjmp(png_jmpbuf(png_ptr)))
    {
      fprintf(stderr, "libpng setjmp failure\n");
      _iu_close_png_from_file(fp, png_ptr, info_ptr);
      return PNG_ERR;
    }

  // Verify png file
  ret = _iu_verify_png(png_ptr, info_ptr, spec);

  // Close png file
  IUReturnCode close_ret = _iu_close_png_from_file(fp, png_ptr, info_ptr);

  return (ret == SUCCESS) ? close_ret : ret;
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

static IUReturnCode
_iu_verify_png_from_memory(IUPngMemReadInfo* readinfo, IUImageSpec* spec,
                           png_structp* png_ptr, png_infop* info_ptr)
{
  int width, height;
  png_byte color_type;

  IUReturnCode ret = SUCCESS;

  if (readinfo->len <= 8)
    {
      fprintf(stderr, "Expected size of at least 8 bytes for PNG\n");
      return IO_ERR;
    }

  if (png_sig_cmp(readinfo->buf, 0, 8))
    {
      fprintf(stderr, "Error reading header\n");
      return IO_ERR;
    }

  *png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!(*png_ptr))
    {
      fprintf(stderr, "png_create_read_struct failed\n");
      return PNG_ERR;
    }

  *info_ptr = png_create_info_struct(*png_ptr);
  if (!(*info_ptr))
    {
      fprintf(stderr, "png_create_info_struct failed\n");
      ret = PNG_ERR;
      goto verify_png_from_memory_destroy_read_struct;
    }

  if (setjmp(png_jmpbuf(*png_ptr)))
    {
      fprintf(stderr, "libpng setjmp failure\n");
      ret = PNG_ERR;
      goto verify_png_from_memory_destroy_info_struct;
    }

  png_set_read_fn(*png_ptr, readinfo, png_mem_read_cb);

  // Skip the header we've already verified
  readinfo->pos = 8;
  png_set_sig_bytes(*png_ptr, 8);

  png_read_info(*png_ptr, *info_ptr);

  width = png_get_image_width(*png_ptr, *info_ptr);
  height = png_get_image_height(*png_ptr, *info_ptr);

  if (_iu_verify_size(width, height, spec) != SUCCESS)
    {
      ret = NON_CONFORMANT;
      goto verify_png_from_memory_destroy_info_struct;
    }

  // Verify this is an 8-bit png we're reading
  color_type = png_get_color_type(*png_ptr, *info_ptr);
  if (color_type != PNG_COLOR_TYPE_GRAY &&
      color_type != PNG_COLOR_TYPE_PALETTE)
    {
      fprintf(stderr, "Expected an 8-bit color type\n");
      ret = NON_CONFORMANT;
      goto verify_png_from_memory_destroy_info_struct;
    }

  if (png_get_bit_depth(*png_ptr, *info_ptr) != 8)
    {
      fprintf(stderr, "Expected 8-bit pixel depth\n");
      ret = NON_CONFORMANT;
      goto verify_png_from_memory_destroy_info_struct;
    }

  return SUCCESS;

verify_png_from_memory_destroy_info_struct:
  png_destroy_info_struct(*png_ptr, info_ptr);
verify_png_from_memory_destroy_read_struct:
  png_destroy_read_struct(png_ptr, NULL, NULL);

  return ret;
}

IUReturnCode
iu_read_png_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec,
                        void** output)
{
  png_structp png_ptr;
  png_infop info_ptr;

  IUPngMemReadInfo readinfo = { (int)len, 0, buffer };
  IUImageSpec blank_spec = {0, 0, 8, 1};

  if (!spec)
    {
      spec = &blank_spec;
    }

  // Verify png file
  IUReturnCode ret = _iu_verify_png_from_memory(&readinfo, spec,
                                                &png_ptr, &info_ptr);
  if (ret != SUCCESS)
    {
      return ret;
    }

  ret = _iu_read_png(&png_ptr, &info_ptr, spec, output);

  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);

  return ret;
}

IUReturnCode
iu_verify_png_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec)
{
  png_structp png_ptr;
  png_infop info_ptr;

  IUPngMemReadInfo readinfo = { (int)len, 0, buffer };
  IUReturnCode ret = _iu_verify_png_from_memory(&readinfo, spec,
                                                &png_ptr, &info_ptr);
  if (ret == SUCCESS)
    {
      png_destroy_info_struct(png_ptr, &info_ptr);
      png_destroy_read_struct(&png_ptr, NULL, NULL);
    }
  return ret;
}

static IUReturnCode
_iu_verify_exr_spec(IUImageSpec* spec)
{
  // TODO: Handle more than 1 channel
  if (spec->channels != 0 && spec->channels != 1)
    {
      fprintf(stderr, "Expected single channel in EXR spec, found %d\n",
              spec->channels);
      return BAD_SPEC;
    }
  spec->channels = 1;

  if (spec->depth != 0 && spec->depth != 16 && spec->depth != 32)
    {
      fprintf(stderr, "Expected 16 or 32-bit depth in EXR spec, found %d\n",
              spec->depth);
      return BAD_SPEC;
    }

  return SUCCESS;
}

static IUReturnCode
_iu_verify_exr_version(int ret, EXRVersion* version)
{
  if (ret != TINYEXR_SUCCESS)
    {
      fprintf(stderr, "Error %02d reading EXR version\n", ret);
      return EXR_ERR;
    }

  if (version->multipart || version->non_image)
    {
      fprintf(stderr, "Can't load multipart or DeepImage EXR image\n");
      return EXR_ERR;
    }

  return SUCCESS;
}

static IUReturnCode
_iu_verify_exr_header(int ret, EXRHeader* header, EXRVersion* version,
                      const char* err, IUImageSpec* spec)
{
  if (ret != TINYEXR_SUCCESS)
    {
      fprintf(stderr, "Error %02d reading EXR header: %s\n", ret, err);
      return EXR_ERR;
    }

  if (spec->channels > 0 && header->num_channels != spec->channels)
    {
      fprintf(stderr, "Expected %d channels, found %d\n",
              spec->channels, header->num_channels);
      FreeEXRHeader(header);
      return NON_CONFORMANT;
    }

  for (int i = 0; i < header->num_channels; i++)
    {
      if (header->channels[i].pixel_type != TINYEXR_PIXELTYPE_HALF &&
          header->channels[i].pixel_type != TINYEXR_PIXELTYPE_FLOAT)
        {
          fprintf(stderr, "Expected all floating-point channels\n");
          FreeEXRHeader(header);
          return NON_CONFORMANT;
        }
    }

  spec->depth = (header->channels[0].pixel_type == TINYEXR_PIXELTYPE_HALF) ?
    16 : 32;

  int width = header->data_window[2] - header->data_window[0] + 1;
  int height = header->data_window[3] - header->data_window[1] + 1;

  if (_iu_verify_size(width, height, spec) != SUCCESS)
    {
      FreeEXRHeader(header);
      return NON_CONFORMANT;
    }

  return SUCCESS;
}

static IUReturnCode
_iu_verify_exr_from_file(const char* filename, IUImageSpec* spec,
                         EXRHeader* header)
{
  IUImageSpec blank_spec = {0, 0, 32, 1};
  if (!spec)
    {
      spec = &blank_spec;
    }
  if (_iu_verify_exr_spec(spec) != SUCCESS)
    {
      return BAD_SPEC;
    }

  EXRVersion version;
  int ret = ParseEXRVersionFromFile(&version, filename);

  if (_iu_verify_exr_version(ret, &version) != SUCCESS)
    {
      return EXR_ERR;
    }

  const char *err = NULL;
  ret = ParseEXRHeaderFromFile(header, &version, filename, &err);

  return _iu_verify_exr_header(ret, header, &version, err, spec);
}

IUReturnCode
iu_read_exr_from_file(const char* filename, IUImageSpec* spec, void** output)
{
  EXRHeader header;

  const char *err = NULL;
  IUImageSpec blank_spec = {0, 0, 32, 1};

  if (!spec)
    {
      spec = &blank_spec;
    }

  IUReturnCode ret = _iu_verify_exr_from_file(filename, spec, &header);
  if (ret != SUCCESS)
    {
      return ret;
    }

  EXRImage exr_image;
  InitEXRImage(&exr_image);

  header.requested_pixel_types[0] = (spec->depth == 16) ?
    TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;

  if (LoadEXRImageFromFile(&exr_image, &header, filename, &err) ==
      TINYEXR_SUCCESS)
    {
      if (!(*output))
        {
          *output = xmalloc(spec->width * spec->height * (spec->depth / 8));
        }

      memcpy(*output,
             &exr_image.images[0][0],
             spec->width * spec->height * (spec->depth / 8));
    }
  else
    {
      ret = EXR_ERR;
    }

  FreeEXRImage(&exr_image);
  FreeEXRHeader(&header);

  return ret;
}

IUReturnCode
iu_verify_exr_from_file(const char* filename, IUImageSpec* spec)
{
  EXRHeader header;
  IUReturnCode ret = _iu_verify_exr_from_file(filename, spec, &header);
  if (ret == SUCCESS)
    {
      FreeEXRHeader(&header);
    }
  return ret;
}

static IUReturnCode
_iu_verify_exr_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec,
                           EXRHeader* header)
{
  IUImageSpec blank_spec = {0, 0, 32, 1};
  if (!spec)
    {
      spec = &blank_spec;
    }
  if (_iu_verify_exr_spec(spec) != SUCCESS)
    {
      return BAD_SPEC;
    }

  EXRVersion version;
  const unsigned char* memory = (const unsigned char*)buffer;
  int ret = ParseEXRVersionFromMemory(&version, memory, len);

  if (_iu_verify_exr_version(ret, &version) != SUCCESS)
    {
      return EXR_ERR;
    }

  const char *err = NULL;
  ret = ParseEXRHeaderFromMemory(header, &version, memory, len, &err);

  return _iu_verify_exr_header(ret, header, &version, err, spec);
}

IUReturnCode
iu_read_exr_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec,
                        void** output)
{
  EXRHeader header;

  const char *err = NULL;
  IUImageSpec blank_spec = {0, 0, 32, 1};

  if (!spec)
    {
      spec = &blank_spec;
    }

  IUReturnCode ret = _iu_verify_exr_from_memory(buffer, len, spec, &header);
  if (ret != SUCCESS)
    {
      return ret;
    }

  EXRImage exr_image;
  InitEXRImage(&exr_image);

  header.requested_pixel_types[0] = (spec->depth == 16) ?
    TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;

  const unsigned char* memory = (const unsigned char*)buffer;
  if (LoadEXRImageFromMemory(&exr_image, &header, memory, len, &err) ==
      TINYEXR_SUCCESS)
    {
      if (!(*output))
        {
          *output = xmalloc(spec->width * spec->height * (spec->depth / 8));
        }

      memcpy(*output,
             &exr_image.images[0][0],
             spec->width * spec->height * (spec->depth / 8));
    }
  else
    {
      ret = EXR_ERR;
    }

  FreeEXRImage(&exr_image);
  FreeEXRHeader(&header);

  return ret;
}

IUReturnCode
iu_verify_exr_from_memory(uint8_t* buffer, size_t len, IUImageSpec* spec)
{
  EXRHeader header;
  IUReturnCode ret = _iu_verify_exr_from_memory(buffer, len, spec, &header);
  if (ret == SUCCESS)
    {
      FreeEXRHeader(&header);
    }
  return ret;
}
