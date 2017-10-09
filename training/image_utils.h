
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

typedef struct {
  int width;    // Expected width, or 0
  int height;   // Expected height, or 0
  int depth;    // Expected depth, per channel, or 0
  int channels; // Expected number of channels (palettised = 1 channel), or 0
} IUImageSpec;

// If output points to a non-NULL address, it is assumed to be pre-allocated,
// otherwise it will be allocated with xmalloc.

// If spec is non-NULL, it will be filled with the specification of the loaded
// file. Any non-zero values are used as file validation.

IUReturnCode iu_read_png_from_file(const char*  filename,
                                   IUImageSpec* spec,
                                   void**       output,
                                   void**       pal_output = NULL,
                                   int*         pal_size = NULL);

IUReturnCode iu_verify_png_from_file(const char*  filename,
                                     IUImageSpec* spec);

IUReturnCode iu_read_png_from_memory(uint8_t*     buffer,
                                     size_t       len,
                                     IUImageSpec* spec,
                                     void**       output,
                                     void**       pal_output = NULL,
                                     int*         pal_size = NULL);

IUReturnCode iu_verify_png_from_memory(uint8_t*     buffer,
                                       size_t       len,
                                       IUImageSpec* spec);

IUReturnCode iu_read_exr_from_file(const char*  filename,
                                   IUImageSpec* spec,
                                   void**       output);

IUReturnCode iu_verify_exr_from_file(const char*  filename,
                                     IUImageSpec* spec);

IUReturnCode iu_read_exr_from_memory(uint8_t*     buffer,
                                     size_t       len,
                                     IUImageSpec* spec,
                                     void**       output);

IUReturnCode iu_verify_exr_from_memory(uint8_t*     buffer,
                                       size_t       len,
                                       IUImageSpec* spec);


#ifdef __cplusplus
};
#endif
