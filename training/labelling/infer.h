
#ifndef __INFER__
#define __INFER__

#include <stdint.h>

float* infer(char**   files,
             uint32_t n_files,
             float*   depth_image,
             uint32_t width,
             uint32_t height,
             uint8_t* n_labels);

#endif /* __INFER__ */
