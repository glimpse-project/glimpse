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


#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SUCCESS,         // Everything went ok
  BAD_SPEC,        // Invalid or unhandled specification
  NON_CONFORMANT,  // File does not conform to given specification
  BAD_FORMAT,      // File is not of the expected format
  IO_ERR,          // File IO error
  PNG_ERR,         // libpng error
  EXR_ERR          // tinyexr error
} IUReturnCode;

typedef enum {
  IU_FORMAT_ANY,
  IU_FORMAT_U8,
  IU_FORMAT_HALF,
  IU_FORMAT_FLOAT,
} IUImageFormat;

typedef struct {
  int width;    // Expected width, or 0
  int height;   // Expected height, or 0

  IUImageFormat format;
} IUImageSpec;

// If output points to a non-NULL address, it is assumed to be pre-allocated,
// otherwise it will be allocated with xmalloc.

// If spec is non-NULL, it will be filled with the specification of the loaded
// file. Any non-zero values are used as file validation.

IUReturnCode iu_read_png_from_file(const char*  filename,
                                   IUImageSpec* spec,
                                   uint8_t**    output,
                                   void**       pal_output,
                                   int*         pal_size);

IUReturnCode iu_read_png_from_memory(uint8_t*     buffer,
                                     size_t       len,
                                     IUImageSpec* spec,
                                     uint8_t**    output,
                                     void**       pal_output,
                                     int*         pal_size);

IUReturnCode iu_verify_png_from_memory(uint8_t*     buffer,
                                       size_t       len,
                                       IUImageSpec* spec);

IUReturnCode iu_write_png_to_file(const char*  filename,
                                  IUImageSpec* spec,
                                  void*        data,
                                  void*        pal,
                                  int          pal_size);

IUReturnCode iu_read_exr_from_file(const char*  filename,
                                   IUImageSpec* spec,
                                   void**       output);

IUReturnCode iu_read_exr_from_memory(uint8_t*     buffer,
                                     size_t       len,
                                     IUImageSpec* spec,
                                     void**       output);

IUReturnCode iu_write_exr_to_file(const char*   filename,
                                  IUImageSpec*  spec,
                                  void*         data,
                                  IUImageFormat format);

const char *iu_code_to_string(IUReturnCode code);

#ifdef __cplusplus
};
#endif
