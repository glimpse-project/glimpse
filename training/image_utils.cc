
#include "image_utils.h"

#include <string.h>
#include <png.h>

#include "tinyexr.h"
#include "half.hpp"
#include "xalloc.h"

using half_float::half;

static IUReturnCode
_iu_verify_png(png_structp png_ptr, png_infop info_ptr, IUImageSpec* spec)
{
  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);

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

IUReturnCode
iu_read_png_from_file(const char* filename, IUImageSpec* spec, char** output)
{
  FILE* fp;
  png_structp png_ptr;
  png_infop info_ptr;
  int row_stride;
  png_bytep* input_rows;
  png_bytep input_data;

  IUImageSpec blank_spec = {0, 0, 8, 1};
  IUReturnCode ret = SUCCESS;

  if (!spec)
    {
      spec = &blank_spec;
    }

  if (spec->depth != 8)
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
  if (ret != SUCCESS)
    {
      _iu_close_png_from_file(fp, png_ptr, info_ptr);
      return ret;
    }

  // Allocate output if necessary
  if (!(*output))
    {
      *output = (char*)xmalloc(spec->width * spec->height);
    }

  // Start reading data
  row_stride = png_get_rowbytes(png_ptr, info_ptr);
  input_rows = (png_bytep *)
    xmalloc(sizeof(png_bytep*) * spec->height);
  input_data = (png_bytep)
    xmalloc(row_stride * spec->height * sizeof(png_bytep));

  for (int y = 0; y < spec->height; y++)
    {
      input_rows[y] = (png_byte*)input_data + row_stride * y;
    }

  png_read_image(png_ptr, input_rows);

  // Copy label image data into training context struct
  for (int y = 0, src_idx = 0, dst_idx = 0;
       y < spec->height; y++, src_idx += row_stride, dst_idx += spec->width)
    {
      memcpy(&((*output)[dst_idx]), &input_data[src_idx], spec->width);
    }

  // Free data associated with PNG reading
  xfree(input_rows);
  xfree(input_data);

  return _iu_close_png_from_file(fp, png_ptr, info_ptr);
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

  if (spec->depth != 8)
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
