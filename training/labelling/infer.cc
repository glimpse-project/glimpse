
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

  // Start inference
  uint8_t n_labels = 0;
  float* output_pr = NULL;
  for (int i = 3; i < argc; i++)
    {
      // Validate the decision tree
      FILE* tree_file = fopen(argv[i], "rb");
      if (!tree_file)
        {
          fprintf(stderr, "Error opening tree '%s'\n", argv[i]);
          return 1;
        }

      RDLHeader header;
      if (fread(&header, sizeof(RDLHeader), 1, tree_file) != 1)
        {
          fprintf(stderr, "Error reading header in '%s'\n", argv[i]);
          return 1;
        }

      if (strncmp(header.tag, "RDT", 3) != 0)
        {
          fprintf(stderr, "'%s' is not an RDT file\n", argv[i]);
          return 1;
        }

      if (header.version != OUT_VERSION)
        {
          fprintf(stderr, "Incompatible RDT version in '%s'\n"
                  "Expected %u, found %u\n", argv[i], OUT_VERSION,
                  (uint32_t)header.version);
          return 1;
        }

      if (n_labels == 0)
        {
          n_labels = header.n_labels;
          output_pr = (float*)
            xcalloc(width * height * n_labels, sizeof(float));
        }
      if (header.n_labels != n_labels)
        {
          fprintf(stderr, "Tree in '%s' has %u labels, expected %u\n",
                  argv[i], (uint32_t)header.n_labels, (uint32_t)n_labels);
          return 1;
        }

      // Read in the decision tree nodes
      uint32_t n_nodes = (uint32_t)roundf(powf(2.f, header.depth)) - 1;
      Node* tree = (Node*)xmalloc(n_nodes * sizeof(Node));
      if (fread(tree, sizeof(Node), n_nodes, tree_file) != n_nodes)
        {
          fprintf(stderr, "Error reading tree nodes in '%s'\n", argv[i]);
          return 1;
        }

      // Read in the label probabilities
      long label_pr_pos = ftell(tree_file);
      fseek(tree_file, 0, SEEK_END);
      long label_bytes = ftell(tree_file) - label_pr_pos;
      if (label_bytes % sizeof(float) != 0)
        {
          fprintf(stderr, "Unexpected size of label probability tables\n");
          return 1;
        }
      uint32_t n_prs = label_bytes / sizeof(float);
      if (n_prs % n_labels != 0)
        {
          fprintf(stderr, "Unexpected number of label probabilities\n");
          return 1;
        }
      uint32_t n_tables = n_prs / n_labels;
      fseek(tree_file, label_pr_pos, SEEK_SET);

      float* label_prs = (float*)xmalloc(label_bytes);
      if (fread(label_prs, sizeof(float) * n_labels,
                n_tables, tree_file) != n_tables)
        {
          fprintf(stderr, "Error reading label probability tables in '%s'\n",
                  argv[i]);
          return 1;
        }

      // Close tree file
      fclose(tree_file);

      // Accumulate probability map
      for (int y = 0; y < height; y++)
        {
          for (int x = 0; x < width; x++)
            {
              Int2D pixel = { x, y };
              float depth_value = depth_image[y * width + x];

              Node* node = tree;
              uint32_t id = 0;
              while (node->label_pr_idx == 0)
                {
                  float value = sample_uv(depth_image, width, height, pixel,
                                          depth_value, node->uv);
                  id = (value < node->t) ? 2 * id + 1 : 2 * id + 2;
                  node = &tree[id];
                }

              float* pr_table = &label_prs[(node->label_pr_idx - 1) * n_labels];
              float* out_pr_table = &output_pr[(y * width * n_labels) +
                                               (x * n_labels)];
              for (int i = 0; i < n_labels; i++)
                {
                  out_pr_table[i] += pr_table[i];
                }
            }
        }

      // Free allocated tree structures
      xfree(tree);
      xfree(label_prs);
    }

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
