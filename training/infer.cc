
#include <stdbool.h>
#include <math.h>

#include "half.hpp"

#include "infer.h"
#include "xalloc.h"
#include "utils.h"
#include "loader.h"

#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.001f

using half_float::half;

float*
infer_labels(RDTree** forest, uint8_t n_trees, half* depth_image,
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

                  /* NB: The nodes are arranged in breadth-first, left then
                   * right child order with the root node at index zero.
                   *
                   * In this case if you have an index for any particular node
                   * ('id' here) then 2 * id + 1 is the index for the left
                   * child and 2 * id + 2 is the index for the right child...
                   */
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

float*
calc_pixel_weights(half* depth_image, float* pr_table,
                   int32_t width, int32_t height, uint8_t n_labels,
                   LList** joint_map, uint8_t n_joints, float* weights)
{
  if (!weights)
    {
      weights = (float*)xmalloc(width * height * n_joints * sizeof(float));
    }

  for (int32_t y = 0, weight_idx = 0, pixel_idx = 0; y < height; y++)
    {
      for (int32_t x = 0; x < width; x++, pixel_idx++)
        {
          float depth_2 = powf((float)depth_image[pixel_idx], 2);

          for (uint8_t j = 0; j < n_joints; j++, weight_idx++)
            {
              float pr = 0.f;
              for (LList* node = joint_map[j]; node; node = node->next)
                {
                  uint8_t label = (uint8_t)((uintptr_t)node->data);
                  pr += pr_table[(pixel_idx * n_labels) + label];
                }
              weights[weight_idx] = pr * depth_2;
            }
        }
    }

  return weights;
}

float*
infer_joints(half* depth_image, float* pr_table, float* weights,
             int32_t width, int32_t height,
             uint8_t n_labels, LList** joint_map, uint8_t n_joints,
             float vfov, JIParam* params)
{
  // Use mean-shift to find the inferred joint positions, set them back into
  // the body using the given offset, and return the results
  uint32_t* n_pixels = (uint32_t*)xcalloc(n_joints, sizeof(uint32_t));
  size_t points_size = n_joints * width * height * 3 * sizeof(float);
  float* points = (float*)xmalloc(points_size);
  float* density = (float*)xmalloc(points_size);

  // Variables for reprojection of 2d point + depth
  float half_width = width / 2.f;
  float half_height = height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = vfov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;
  //float hfov = atanf(tan_half_hfov) * 2;

  float root_2pi = sqrtf(2 * M_PI);

  // Gather pixels above the given threshold
  for (int32_t y = 0, idx = 0; y < height; y++)
    {
      float t = -((y / half_height) - 1.f);
      for (int32_t x = 0; x < width; x++, idx++)
        {
          float s = (x / half_width) - 1.f;
          float depth = (float)depth_image[idx];
          if (!std::isnormal(depth))
            {
              continue;
            }

          for (uint8_t j = 0; j < n_joints; j++)
            {
              float threshold = params[j].threshold;
              uint32_t joint_idx = j * width * height;
              for (LList* node = joint_map[j]; node; node = node->next)
                {
                  uint8_t label = (uint8_t)((uintptr_t)node->data);
                  float label_pr = pr_table[(idx * n_labels) + label];
                  if (label_pr >= threshold)
                    {
                      // Reproject point
                      points[(joint_idx + n_pixels[j]) * 3] =
                        (tan_half_hfov * depth) * s;
                      points[(joint_idx + n_pixels[j]) * 3 + 1] =
                        (tan_half_vfov * depth) * t;
                      points[(joint_idx + n_pixels[j]) * 3 + 2] =
                        depth;

                      // Store pixel weight (density)
                      density[joint_idx + n_pixels[j]] =
                        weights[(idx * n_joints) + j];

                      n_pixels[j]++;
                      break;
                    }
                }
            }
        }
    }

  float* joints = (float*)xcalloc(n_joints * 3, sizeof(float));

  // Means shift to find joint modes
  for (uint8_t j = 0; j < n_joints; j++)
    {
      if (n_pixels[j] == 0)
        {
          continue;
        }

      float bandwidth = params[j].bandwidth;
      float offset = params[j].offset;

      uint32_t joint_idx = j * width * height;
      for (uint32_t s = 0; s < N_SHIFTS; s++)
        {
          float new_points[n_pixels[j] * 3];
          bool moved = false;
          for (uint32_t p = 0; p < n_pixels[j]; p++)
            {
              float* x = &points[(joint_idx + p) * 3];
              float* nx = &new_points[p * 3];
              float numerator[3] = { 0.f, };
              float denominator = 0.f;
              for (uint32_t n = 0; n < n_pixels[j]; n++)
                {
                  float* xi = &points[(joint_idx + n) * 3];
                  float distance = sqrtf(pow(x[0] - xi[0], 2.f) +
                                         pow(x[1] - xi[1], 2.f) +
                                         pow(x[2] - xi[2], 2.f));

                  // Weighted gaussian kernel
                  float weight = density[joint_idx + n] *
                    (1.f / (bandwidth * root_2pi)) *
                    expf(-0.5f * pow(distance / bandwidth, 2.f));

                  numerator[0] += weight * xi[0];
                  numerator[1] += weight * xi[1];
                  numerator[2] += weight * xi[2];

                  denominator += weight;
                }

              nx[0] = numerator[0] / denominator;
              nx[1] = numerator[1] / denominator;
              nx[2] = numerator[2] / denominator;

              if (!moved &&
                  (fabs(nx[0] - x[0]) >= SHIFT_THRESHOLD ||
                   fabs(nx[1] - x[1]) >= SHIFT_THRESHOLD ||
                   fabs(nx[2] - x[2]) >= SHIFT_THRESHOLD))
                {
                  moved = true;
                }
            }

          memcpy((void*)&points[joint_idx * 3], (void*)new_points,
                 n_pixels[j] * 3 * sizeof(float));

          if (!moved || s == N_SHIFTS - 1)
            {
              // Find the mode we're most confident of
              float* last_point = &points[joint_idx * 3];
              float confidence = 0;

              float* best_point = last_point;
              float best_confidence = 0;

              //uint32_t unique_points = 1;

              for (uint32_t p = 0; p < n_pixels[j]; p++)
                {
                  float* point = &points[(joint_idx + p) * 3];
                  if (fabs(point[0]-last_point[0]) >= SHIFT_THRESHOLD ||
                      fabs(point[1]-last_point[1]) >= SHIFT_THRESHOLD ||
                      fabs(point[2]-last_point[2]) >= SHIFT_THRESHOLD)
                    {
                      if (confidence > best_confidence)
                        {
                          best_point = last_point;
                          best_confidence = confidence;
                        }
                      //unique_points++;
                      last_point = point;
                      confidence = 0;
                    }
                  confidence += density[joint_idx + p];
                }

              // Offset into the body
              best_point[2] += offset;

              // Store joint
              joints[j * 3] = best_point[0];
              joints[j * 3 + 1] = best_point[1];
              joints[j * 3 + 2] = best_point[2];

              break;
            }
        }
    }

  xfree(density);
  xfree(points);
  xfree(n_pixels);

  return joints;
}
