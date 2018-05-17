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

#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include <png.h>

#include "half.hpp"

#include <glimpse_log.h>

#include "image_utils.h"
#include "xalloc.h"
#include "utils.h"
#include "loader.h"
#include "infer.h"

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

using half_float::half;

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
  struct gm_logger *log = gm_logger_new(NULL, NULL);
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
  half* depth_image = NULL;
  IUImageSpec exr_spec = { 0, 0, IU_FORMAT_HALF };
  if (iu_read_exr_from_file(argv[optind], &exr_spec,
                            (void**)(&depth_image)) != SUCCESS)
    {
      fprintf(stderr, "Error loading depth image\n");
      return 1;
    }
  int width = exr_spec.width;
  int height = exr_spec.height;

  if (!write_palettized_pngs)
    {
      palette = NULL;
    }
  else if (palette_file)
    {
      palette = NULL;
      if (iu_read_png_from_file(palette_file, NULL, NULL,
                                (void**)(&palette), &palette_size) != SUCCESS)
        {
          fprintf(stderr, "Failed to read palette\n");
          return 1;
        }
    }

  // Do inference
  unsigned n_trees = argc - optind - 2;
  RDTree** forest = read_forest(log, (const char**)&argv[optind+2], n_trees, NULL);
  if (!forest)
    {
      return 1;
    }
  float* output_pr = infer_labels<half>(forest, n_trees, depth_image,
                                        width, height);
  uint8_t n_labels = forest[0]->header.n_labels;
  free_forest(forest, n_trees);

  // Write out png of most likely labels
  png_bytep out_labels = (png_bytep)xmalloc(width * height);
  for (int y = 0; y < height; y++)
    {
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

  IUImageSpec output_spec = { width, height, IU_FORMAT_U8 };
  if (iu_write_png_to_file(argv[optind+1], &output_spec,
                           out_labels, palette, palette_size) != SUCCESS)
    {
      fprintf(stderr, "Error writing output PNG\n");
    }

  // Clean-up
  xfree(out_labels);
  xfree(output_pr);
  xfree(depth_image);

  return 0;
}
