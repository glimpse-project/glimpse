#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include <png.h>

#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImathBox.h>

#include "half.hpp"

#include "xalloc.h"
#include "utils.h"
#include "loader.h"
#include "infer.h"

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static png_color default_palette[] = {
    { 0xff, 0x5d, 0xaa },
    { 0xd1, 0x15, 0x40 },
    { 0xda, 0x1d, 0x0e },
    { 0xdd, 0x5d, 0x1e },
    { 0x49, 0xa2, 0x24 },
    { 0x29, 0xdc, 0xe3 },
    { 0x02, 0x68, 0xc2 },
    { 0x90, 0x29, 0xf9 },
    { 0xff, 0x00, 0xcf },
    { 0xef, 0xd2, 0x37 },
    { 0x92, 0xa1, 0x3a },
    { 0x48, 0x21, 0xeb },
    { 0x2f, 0x93, 0xe5 },
    { 0x1d, 0x6b, 0x0e },
    { 0x07, 0x66, 0x4b },
    { 0xfc, 0xaa, 0x98 },
    { 0xb6, 0x85, 0x91 },
    { 0xab, 0xae, 0xf1 },
    { 0x5c, 0x62, 0xe0 },
    { 0x48, 0xf7, 0x36 },
    { 0xa3, 0x63, 0x0d },
    { 0x78, 0x1d, 0x07 },
    { 0x5e, 0x3c, 0x00 },
    { 0x9f, 0x9f, 0x60 },
    { 0x51, 0x76, 0x44 },
    { 0xd4, 0x6d, 0x46 },
    { 0xff, 0xfb, 0x7e },
    { 0xd8, 0x4b, 0x4b },
    { 0xa9, 0x02, 0x52 },
    { 0x0f, 0xc1, 0x66 },
    { 0x2b, 0x5e, 0x44 },
    { 0x00, 0x9c, 0xad },
    { 0x00, 0x40, 0xad },
    { 0x21, 0x21, 0x21 },
};

static png_color *palette = default_palette;
static int palette_size = ARRAY_LEN(default_palette);

static void
load_palette_from_png(const char *palette_png)
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
        fprintf(stderr, "Palette file does not have a palette\n");
        exit(1);
      }

    if (png_get_bit_depth(pal_png_ptr, pal_info_ptr) != 8)
      {
        fprintf(stderr, "Palette file is not 8bpp\n");
        exit(1);
      }

    // Read out palette data
    if (!png_get_PLTE(pal_png_ptr, pal_info_ptr, &palette, &palette_size))
      {
        fprintf(stderr, "Error reading palette from palette png\n");
        exit(1);
      }

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

static bool
write_png_file(const char *filename,
               int width, int height,
               png_bytep *row_pointers,
               png_byte color_type)
{
    png_structp png_ptr;
    png_infop info_ptr;
    bool ret = false;

    /* create file */
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return false;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
      {
        fprintf(stderr, "png_create_write_struct failed\n");
        goto error_create_write;
      }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
      {
        fprintf(stderr, "png_create_info_struct failed\n");
        goto error_create_info;
      }

    if (setjmp(png_jmpbuf(png_ptr)))
      {
        fprintf(stderr, "PNG write failure\n");
        goto error_write;
      }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_PLTE(png_ptr, info_ptr, palette, palette_size);

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
"Usage: depth2labels [OPTIONS] <in.exr> <out.png> <tree1.rdt> [tree2.rdt] ...\n"
"Use 1 or more randomised decision trees to infer labelling of a depth image.\n"
"\n"
"  -g, --grey               Write greyscale not palletized label PNGs\n"
"  -p, --palette=PNG_FILE   Use this PNG file's palette instead of default\n"
"\n"
"  -h, --help               Display this help\n\n");
  exit(1);
}

int
main(int argc, char **argv)
{
  bool write_palettized_pngs = true;
  const char *palette_file = NULL;
  int opt;

  /* N.B. The initial '+' means that getopt will stop looking for options
   * after the first non-option argument...
   */
  const char *short_options="+hgp:";
  const struct option long_options[] = {
      {"help",            no_argument,        0, 'h'},
      {"grey",            no_argument,        0, 'g'},
      {"palette",         required_argument,  0, 'p'},
      {0, 0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
         != -1)
    {
      switch (opt) {
          case 'h':
              print_usage(stderr);
              return 0;
          case 'g':
              write_palettized_pngs = false;
              break;
          case 'p':
              palette_file = optarg;
              break;
         default:
              print_usage(stderr);
              return 1;
      }
  }

  if ((argc - optind) < 3)
    {
      print_usage(stderr);
    }

  // Read depth file
  InputFile in_file(argv[optind]);
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

  half_float::half* depth_image =
      (half_float::half*)xmalloc(width * height * sizeof(half_float::half));

  FrameBuffer frameBuffer;
  frameBuffer.insert("Y",
                     Slice (HALF,
                            (char *)depth_image,
                            sizeof(half_float::half), // x stride,
                            sizeof(half_float::half) * width)); // y stride

  in_file.setFrameBuffer(frameBuffer);
  in_file.readPixels(dw.min.y, dw.max.y);

  if (palette_file)
      load_palette_from_png(palette_file);

  // Do inference
  unsigned n_trees = argc - optind - 2;
  RDTree** forest = read_forest(&argv[optind+2], n_trees);
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

  write_png_file(argv[optind+1], width, height, rows,
                 write_palettized_pngs ? PNG_COLOR_TYPE_PALETTE :
                 PNG_COLOR_TYPE_GRAY);

  // Clean-up
  xfree(rows);
  xfree(out_labels);
  xfree(output_pr);
  xfree(depth_image);

  return 0;
}
