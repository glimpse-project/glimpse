
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <png.h>

#include <half.h>
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
               png_byte bit_depth,
               char* palette_png)
{
  png_structp png_ptr;
  png_infop info_ptr;
  bool ret = false;

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    {
      fprintf(stderr, "png_create_write_struct faile\nd");
      goto error_create_write;
    }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    {
      fprintf(stderr, "png_create_info_struct failed");
      goto error_create_info;
    }

  if (setjmp(png_jmpbuf(png_ptr)))
    {
      fprintf(stderr, "PNG write failure");
      goto error_write;
    }

  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr, width, height,
               bit_depth, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
      FILE* pal_fp;
      unsigned char header[8]; // 8 is the maximum size that can be checked
      png_structp pal_png_ptr;
      png_infop pal_info_ptr;

      // Open palette png file
      if (!(pal_fp = fopen(palette_png, "rb")))
        {
          fprintf(stderr, "Failed to open palette image\n");
          exit(1);
        }

      // Load palette png file
      if (fread(header, 1, 8, pal_fp) != 8)
        {
          fprintf(stderr, "Error reading header of palette image\n");
          exit(1);
        }
      if (png_sig_cmp(header, 0, 8))
        {
          fprintf(stderr, "Palette png was not recognised as a PNG file\n");
          exit(1);
        }

      // Start reading palette png
      pal_png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!pal_png_ptr)
        {
          fprintf(stderr, "png_create_read_struct failed\n");
          exit(1);
        }

      pal_info_ptr = png_create_info_struct(pal_png_ptr);
      if (!pal_info_ptr)
        {
          fprintf(stderr, "png_create_info_struct failed\n");
          exit(1);
        }

      if (setjmp(png_jmpbuf(pal_png_ptr)))
        {
          fprintf(stderr, "libpng setjmp failure\n");
          exit(1);
        }

      png_init_io(pal_png_ptr, pal_fp);
      png_set_sig_bytes(pal_png_ptr, 8);

      png_read_info(pal_png_ptr, pal_info_ptr);

      // Verify pixel type
      png_byte color_type = png_get_color_type(pal_png_ptr, pal_info_ptr);
      if (color_type != PNG_COLOR_TYPE_PALETTE)
        {
          fprintf(stderr, "Palette file does not have a pallete\n");
          exit(1);
        }

      if (png_get_bit_depth(pal_png_ptr, pal_info_ptr) != 8)
        {
          fprintf(stderr, "Palette file is not 8bpp\n");
          exit(1);
        }

      // Read out palette data
      png_colorp palette;
      int palette_size;
      if (!png_get_PLTE(pal_png_ptr, pal_info_ptr, &palette, &palette_size))
        {
          fprintf(stderr, "Error reading palette from palette png\n");
          exit(1);
        }

      // Write palette
      png_set_PLTE(png_ptr, info_ptr, palette, palette_size);

      // Close the palette file
      if (fclose(pal_fp) != 0)
        {
          fprintf(stderr, "Error closing palette file\n");
          exit(1);
        }

      // Free data associated with PNG reading
      png_destroy_info_struct(pal_png_ptr, &pal_info_ptr);
      png_destroy_read_struct(&pal_png_ptr, NULL, NULL);
    }

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
"Usage: train [OPTIONS] <in.exr> <out.png> <tree1.rdt> [tree2.rdt] ...\n"
"Use 1 or more randomised decision trees to infer labelling of a depth image.\n"
"\n"
"  -p, --palette=PNG_FILE   Use this PNG file's palette for output\n");
}

int
main(int argc, char **argv)
{
  if (argc < 4)
    {
      print_usage(stderr);
      return 1;
    }

  char** arguments = &argv[1];
  char* palette = NULL;
  if (strcmp(argv[1], "-p") == 0)
    {
      if (argc < 6)
        {
          print_usage(stderr);
          return 1;
        }
      palette = argv[2];
      arguments = &argv[3];
    }
  else if (strncmp(argv[1], "--palette=", 10) == 0)
    {
      if (argc < 5)
        {
          print_usage(stderr);
          return 1;
        }
      palette = &argv[1][10];
      arguments = &argv[2];
    }

  // Read depth file
  InputFile in_file(arguments[0]);
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

  half* depth_image = (half*)xmalloc(width * height * sizeof(half));

  FrameBuffer frameBuffer;
  frameBuffer.insert("Y",
                     Slice (HALF,
                            (char *)depth_image,
                            sizeof(half), // x stride,
                            sizeof(half) * width)); // y stride

  in_file.setFrameBuffer(frameBuffer);
  in_file.readPixels(dw.min.y, dw.max.y);

  // Open output file
  FILE* out_png_file = fopen(arguments[1], "wb");
  if (!out_png_file)
    {
      fprintf(stderr, "Error opening output file\n");
    }

  // Do inference
  uint8_t n_trees = argc - (arguments - argv) - 2;
  RDTree** forest = read_forest(&arguments[2], n_trees);
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

  write_png_file(out_png_file, width, height, rows,
                 palette ? PNG_COLOR_TYPE_PALETTE : PNG_COLOR_TYPE_GRAY, 8,
                 palette);

  // Clean-up
  xfree(rows);
  xfree(out_labels);
  xfree(output_pr);
  xfree(depth_image);

  return 0;
}
