
#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SUCCESS,
  BAD_SPEC,
  NON_CONFORMANT,
  BAD_FORMAT,
  IO_ERR,
  PNG_ERR,
  EXR_ERR
} IUReturnCode;

typedef struct {
  int width;
  int height;
  int depth;
  int channels;
} IUImageSpec;

// If output points to a non-NULL address, it is assumed to be pre-allocated.
IUReturnCode iu_read_png_from_file(const char* filename,
                                   IUImageSpec* spec,
                                   void** output);

IUReturnCode iu_verify_png_from_file(const char* filename,
                                     IUImageSpec* spec);

IUReturnCode iu_read_exr_from_file(const char* filename,
                                   IUImageSpec* spec,
                                   void** output);

IUReturnCode iu_verify_exr_from_file(const char* filename,
                                     IUImageSpec* spec);

IUReturnCode iu_read_exr_from_memory(uint8_t* buffer,
                                     size_t len,
                                     IUImageSpec* spec,
                                     void** output);

IUReturnCode iu_verify_exr_from_memory(uint8_t* buffer,
                                       size_t len,
                                       IUImageSpec* spec);


#ifdef __cplusplus
};
#endif
