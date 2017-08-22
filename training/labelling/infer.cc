
#include "infer.h"
#include "xalloc.h"
#include "utils.h"
#include "loader.h"

float*
infer(RDTree** forest, uint8_t n_trees, float* depth_image,
      uint32_t width, uint32_t height)
{
  uint8_t n_labels = forest[0]->header.n_labels;
  float* output_pr = (float*)
    xcalloc(width * height * forest[0]->header.n_labels, sizeof(float));

  for (uint8_t i = 0; i < n_trees; i++)
    {
      RDTree* tree = forest[i];

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
    }

  // Correct the probabilities
  for (uint32_t y = 0, idx = 0; y < height; y++)
    {
      for (uint32_t x = 0; x < width; x++)
        {
          for (uint8_t l = 0; l < n_labels; l++, idx++)
            {
              output_pr[idx] /= (float)n_trees;
            }
        }
    }

  return output_pr;
}
