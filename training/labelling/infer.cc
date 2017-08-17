
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "infer.h"
#include "xalloc.h"
#include "utils.h"
#include "loader.h"

float*
infer(char** files, uint32_t n_files, float* depth_image,
      uint32_t width, uint32_t height, uint8_t* out_n_labels)
{
  bool error = false;
  float* output_pr = NULL;
  uint8_t n_labels = 0;

  for (uint32_t i = 0; i < n_files; i++)
    {
      // Validate the decision tree
      FILE* tree_file = fopen(files[i], "rb");
      if (!tree_file)
        {
          fprintf(stderr, "Error opening tree '%s'\n", files[i]);
          error = true;
          break;
        }
      RDTree* tree = read_rdt(tree_file);
      fclose(tree_file);

      if (!tree)
        {
          error = true;
          break;
        }

      if (n_labels == 0)
        {
          n_labels = tree->header.n_labels;
          output_pr = (float*)
            xcalloc(width * height * n_labels, sizeof(float));
        }
      if (tree->header.n_labels != n_labels)
        {
          fprintf(stderr, "Tree in '%s' has %u labels, expected %u\n",
                  files[i], (uint32_t)tree->header.n_labels,
                  (uint32_t)n_labels);
          error = true;
          break;
        }

      // Accumulate probability map
      for (uint32_t y = 0; y < height; y++)
        {
          for (uint32_t x = 0; x < width; x++)
            {
              Int2D pixel = { (int32_t)x, (int32_t)y };
              float depth_value = depth_image[y * width + x];

              Node* node = tree->nodes;
              uint32_t id = 0;
              while (node->label_pr_idx == 0)
                {
                  float value = sample_uv(depth_image, width, height, pixel,
                                          depth_value, node->uv);
                  id = (value < node->t) ? 2 * id + 1 : 2 * id + 2;
                  node = &tree->nodes[id];
                }

              float* pr_table =
                &tree->label_pr_tables[(node->label_pr_idx - 1) * n_labels];
              float* out_pr_table = &output_pr[(y * width * n_labels) +
                                               (x * n_labels)];
              for (int i = 0; i < n_labels; i++)
                {
                  out_pr_table[i] += pr_table[i];
                }
            }
        }

      // Free RDT
      free_rdt(tree);
    }

  if (error)
    {
      if (output_pr)
        {
          xfree(output_pr);
        }
      return NULL;
    }

  // Correct the probabilities
  for (uint32_t y = 0, idx = 0; y < height; y++)
    {
      for (uint32_t x = 0; x < width; x++)
        {
          for (uint8_t l = 0; l < n_labels; l++, idx++)
            {
              output_pr[idx] /= (float)n_files;
            }
        }
    }

  if (out_n_labels)
    {
      *out_n_labels = n_labels;
    }

  return output_pr;
}
