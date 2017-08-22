
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <png.h>

#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImathBox.h>

#include "xalloc.h"
#include "utils.h"
#include "loader.h"
#include "infer.h"

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static bool
write_png_file(FILE* fp,
               int width, int height,
               png_bytep *row_pointers,
               png_byte color_type,
               png_byte bit_depth)
{
    png_structp png_ptr;
    png_infop info_ptr;
    bool ret = false;

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "png_create_write_struct faile\nd");
        goto error_create_write;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "png_create_info_struct failed");
        goto error_create_info;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "PNG write failure");
        goto error_write;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_write_image(png_ptr, row_pointers);

    png_write_end(png_ptr, NULL);

    ret = true;

error_write:
    png_destroy_info_struct(png_ptr, &info_ptr);
error_create_info:
    png_destroy_write_struct(&png_ptr, NULL);
error_create_write:
    fclose(fp);

    return ret;
}

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: train <in_exr_file> <out_png_file> <tree_file_1> [tree_file_2] ...\n"
"Use 1 or more randomised decision trees to infer labelling of a depth image.\n");
}

int
main(int argc, char **argv)
{
  if (argc < 4)
    {
      print_usage(stderr);
      return 1;
    }

  // Read depth file
  InputFile in_file(argv[1]);
  Box2i dw = in_file.header().dataWindow();

  int32_t width = dw.max.x - dw.min.x + 1;
  int32_t height = dw.max.y - dw.min.y + 1;

  const ChannelList& channels = in_file.header().channels();
  if (!channels.findChannel("Y"))
    {
      fprintf(stderr, "Only expected to load greyscale EXR files "
              "with a single 'Y' channel\n");
      return 1;
    }

  float* depth_image = (float*)xmalloc(width * height * sizeof(float));

  FrameBuffer frameBuffer;
  frameBuffer.insert("Y",
                     Slice (FLOAT,
                            (char *)depth_image,
                            sizeof(float), // x stride,
                            sizeof(float) * width)); // y stride

  in_file.setFrameBuffer(frameBuffer);
  in_file.readPixels(dw.min.y, dw.max.y);

  // Open output file
  FILE* out_png_file = fopen(argv[2], "wb");
  if (!out_png_file)
    {
      fprintf(stderr, "Error opening output file '%s'\n", argv[2]);
    }

  // Do inference
  uint8_t n_trees = argc - 3;
  RDTree** forest = read_forest(&argv[3], n_trees);
  if (!forest)
    {
      return 1;
    }
  float* output_pr = infer(forest, n_trees, depth_image, width, height);
  uint8_t n_labels = forest[0]->header.n_labels;
  free_forest(forest, n_trees);

  // Write out png of most likely labels
  png_bytep* rows = (png_bytep*)xmalloc(height * sizeof(png_bytep));
  png_bytep out_labels = (png_bytep)xmalloc(width * height);
  for (int y = 0; y < height; y++)
    {
      rows[y] = (png_byte*)out_labels + width * y;
      for (int x = 0; x < width; x++)
        {
          uint8_t label = 0;
          float pr = 0.0;
          float* pr_table = &output_pr[(y * width * n_labels) +
                                       (x * n_labels)];
          for (uint8_t l = 0; l < n_labels; l++)
            {
              if (pr_table[l] > pr)
                {
                  label = l;
                  pr = pr_table[l];
                }
            }

          out_labels[y * width + x] = label;
        }
    }

  write_png_file(out_png_file, width, height, rows, PNG_COLOR_TYPE_GRAY, 8);

  // Clean-up
  xfree(rows);
  xfree(out_labels);
  xfree(output_pr);
  xfree(depth_image);

  return 0;
}
