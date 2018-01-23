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


#include <stdbool.h>
#include <math.h>
#include <vector>
#include <list>
#include <forward_list>

#include "half.hpp"

#include "infer.h"
#include "xalloc.h"
#include "utils.h"
#include "loader.h"

#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.01f

#define ARRAY_LEN(ARRAY) (sizeof(ARRAY)/sizeof(ARRAY[0]))


using half_float::half;


typedef struct {
    int n_labels;

    /* XXX: There's a runtime check that we don't have joints mapped to
     * more than two labels and this can easily be bumped if necessary.
     * Keeping this at the minimum size seems likely to help improve
     * locality considering the frequency of accessing this data within
     * some of the inner loops below.
     */
    uint8_t labels[2];
} JointMapEntry;


template<typename FloatT>
float*
infer_labels(RDTree** forest, uint8_t n_trees, FloatT* depth_image,
             uint32_t width, uint32_t height, float* out_labels)
{
  uint8_t n_labels = forest[0]->header.n_labels;

  size_t output_size = width * height *
                       forest[0]->header.n_labels * sizeof(float);
  float* output_pr = out_labels ? out_labels : (float*)xmalloc(output_size);
  memset(output_pr, 0, output_size);

  // Accumulate probability map
  for (uint32_t y = 0; y < height; y++)
    {
      for (uint32_t x = 0; x < width; x++)
        {
          float* out_pr_table = &output_pr[(y * width * n_labels) +
                                           (x * n_labels)];
          float depth_value = (float)depth_image[y * width + x];

          // TODO: Provide a configurable threshold here?
          if (depth_value >= HUGE_DEPTH)
            {
              out_pr_table[forest[0]->header.bg_label] += 1.0f;
              continue;
            }

          Int2D pixel = { (int32_t)x, (int32_t)y };
          for (uint8_t i = 0; i < n_trees; ++i)
            {
              RDTree* tree = forest[i];
              Node* node = tree->nodes;

              uint32_t id = 0;
              while (node->label_pr_idx == 0)
                {
                  float value = sample_uv<FloatT>(depth_image, width, height,
                                                  pixel, depth_value, node->uv);

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

              /* NB: node->label_pr_idx is a base-one index since index zero
               * is reserved to indicate that the node is not a leaf node
               */
              float* pr_table =
                &tree->label_pr_tables[(node->label_pr_idx - 1) * n_labels];
              for (int n = 0; n < n_labels; ++n)
                {
                  out_pr_table[n] += pr_table[n];
                }
            }

          for (int n = 0; n < n_labels; ++n)
            {
              out_pr_table[n] /= (float)n_trees;
            }
        }
    }

  return output_pr;
}

template float*
infer_labels<half>(RDTree**, uint8_t, half*, uint32_t, uint32_t, float*);
template float*
infer_labels<float>(RDTree**, uint8_t, float*, uint32_t, uint32_t, float*);

/* We don't want to be making lots of function calls or dereferencing
 * lots of pointers while accessing the joint map within inner loops
 * so this lets us temporarily unpack the label mappings into a
 * tight array of JointMapEntries.
 */
static inline void
unpack_joint_map(JSON_Value *joint_map, JointMapEntry *map, int n_joints)
{
  for (int i = 0; i < n_joints; i++)
    {
      JSON_Object *entry = json_array_get_object(json_array(joint_map), i);
      JSON_Array *labels = json_object_get_array(entry, "labels");
      unsigned n_labels = json_array_get_count(labels);

      if (n_labels > ARRAY_LEN(map[0].labels))
        {
          fprintf(stderr, "Didn't expect joint to be mapped to > 2 labels\n");
          exit(1);
        }

      map[i].n_labels = n_labels;
      for (unsigned n = 0; n < n_labels; n++)
        {
          map[i].labels[n] = json_array_get_number(labels, n);
        }
    }
}

template<typename FloatT>
float*
calc_pixel_weights(FloatT* depth_image, float* pr_table,
                   int32_t width, int32_t height, uint8_t n_labels,
                   JSON_Value* joint_map, float* weights)
{
  int n_joints = json_array_get_count(json_array(joint_map));

  JointMapEntry map[n_joints];
  unpack_joint_map(joint_map, map, n_joints);

  if (!weights)
    {
      weights = (float*)xmalloc(width * height * n_joints * sizeof(float));
    }

  for (int32_t y = 0, weight_idx = 0, pixel_idx = 0; y < height; y++)
    {
      for (int32_t x = 0; x < width; x++, pixel_idx++)
        {
          float depth = (float)depth_image[pixel_idx];
          float depth_2 = depth * depth;

          for (uint8_t j = 0; j < n_joints; j++, weight_idx++)
            {
              float pr = 0.f;
              for (int n = 0; n < map[j].n_labels; n++)
                {
                  uint8_t label = map[j].labels[n];
                  pr += pr_table[(pixel_idx * n_labels) + label];
                }
              weights[weight_idx] = pr * depth_2;
            }
        }
    }

  return weights;
}

template float*
calc_pixel_weights<half>(half*, float*, int32_t, int32_t, uint8_t,
                         JSON_Value*, float*);
template float*
calc_pixel_weights<float>(float*, float*, int32_t, int32_t, uint8_t,
                          JSON_Value*, float*);

static int
compare_joints(LList* a, LList* b, void* userdata)
{
  Joint* ja = (Joint*)a->data;
  Joint* jb = (Joint*)b->data;
  return ja->confidence - jb->confidence;
}

template<typename FloatT>
InferredJoints*
infer_joints_fast(FloatT* depth_image, float* pr_table, float* weights,
                  int32_t width, int32_t height, uint8_t n_labels,
                  JSON_Value* joint_map, float vfov, JIParam* params)
{
  int n_joints = json_array_get_count(json_array(joint_map));
  JointMapEntry map[n_joints];
  unpack_joint_map(joint_map, map, n_joints);

  // Plan: For each scan-line, scan along and record clusters on 1 dimension.
  //       For each scanline segment, check to see if it intersects with any
  //       segment on neighbouring scanlines, then join the lists together.
  //       Finally, we should have a list of lists of clusters, which we
  //       can then calculate confidence for, then when we have the highest
  //       confidence for each label, we can project the points and calculate
  //       the center-point.
  //
  //       TODO: Let this take a distance so that clusters don't need to be
  //             perfectly contiguous?
  //       TODO: Figure out a way to divide clusters that are only loosely
  //             connected?
  typedef struct {
    int32_t y;
    int32_t left;
    int32_t right;
  } ScanlineSegment;

  typedef std::forward_list<ScanlineSegment> Cluster;
  typedef std::move_iterator<typename Cluster::iterator> ClusterMoveIterator;

  typename std::list<Cluster> clusters[n_joints];

  // Collect clusters across scanlines
  ScanlineSegment* last_segment[n_joints];
  for (int32_t y = 0; y < height; ++y)
    {
      memset(last_segment, 0, sizeof(ScanlineSegment*) * n_joints);
      for (int32_t x = 0; x < width; ++x)
        {
          for (int32_t j = 0; j < n_joints; ++j)
            {
              bool threshold_passed = false;
              for (int n = 0; n < map[j].n_labels; ++n)
                {
                  uint8_t label = map[j].labels[n];
                  float label_pr = pr_table[(y * width + x) * n_labels + label];
                  if (label_pr >= params[j].threshold)
                    {
                      threshold_passed = true;
                      break;
                    }
                }

              if (threshold_passed)
                {
                  // Check to see if this pixel can be added to an existing
                  // cluster.
                  if (last_segment[j] && last_segment[j]->right == x - 1)
                    {
                      last_segment[j]->right = x;
                    }
                  else
                    {
                      clusters[j].emplace_front();
                      clusters[j].front().push_front({ y, x, x });
                      last_segment[j] = &clusters[j].front().front();
                    }
                }
              else
                {
                  last_segment[j] = nullptr;
                }
            }
        }
    }

  // Now iteratively connect the scanline clusters
  for (int j = 0; j < n_joints; j++)
    {
      for (typename std::list<Cluster>::iterator it = clusters[j].begin();
           it != clusters[j].end(); ++it)
        {
          Cluster& parent = *it;
          for (typename std::list<Cluster>::iterator it2 = clusters[j].begin();
               it2 != clusters[j].end();)
            {
              Cluster& candidate = *it2;
              if (it == it2)
                {
                  ++it2;
                  continue;
                }

              // If this scanline segment connects to the cluster being
              // checked, remove it from the cluster list, add it to the
              // checked cluster and break out.
              bool local_change = false;
              for (typename Cluster::iterator p_it = parent.begin();
                   p_it != parent.end() && !local_change; ++p_it)
                {
                  ScanlineSegment& p_segment = *p_it;
                  for (typename Cluster::iterator c_it = candidate.begin();
                       c_it != candidate.end(); ++c_it)
                    {
                      ScanlineSegment& c_segment = *c_it;
                      // Check if these two scanline cluster segments touch
                      if ((abs(c_segment.y - p_segment.y) == 1) &&
                          (c_segment.left <= p_segment.right) &&
                          (c_segment.right >= p_segment.left))
                        {
                          parent.insert_after(
                              parent.before_begin(),
                              ClusterMoveIterator(candidate.begin()),
                              ClusterMoveIterator(candidate.end()));
                          it2 = clusters[j].erase(it2);
                          local_change = true;
                          break;
                        }
                    }
                }

              if (!local_change)
                {
                  ++it2;
                }
            }
        }
    }

  // clusters now contains the boundaries per scanline segment of each cluster
  // of joint labels, which we can now use to calculate the highest confidence
  // cluster and the projected cluster centroid.

  // Variables for reprojection of 2d point + depth
  float half_width = width / 2.f;
  float half_height = height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = vfov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;
  //float hfov = atanf(tan_half_hfov) * 2.f;

  //float root_2pi = sqrtf(2.f * M_PI);

  // Allocate/clear joints structure
  InferredJoints* result = (InferredJoints*)xmalloc(sizeof(InferredJoints));
  result->n_joints = n_joints;
  result->joints = (LList**)xcalloc(n_joints, sizeof(LList*));

  for (int j = 0; j < n_joints; j++)
    {
      for (typename std::list<Cluster>::iterator it =
           clusters[j].begin(); it != clusters[j].end(); ++it)
        {
          Cluster& cluster = *it;
          Joint* joint = (Joint*)xmalloc(sizeof(Joint));

          // Calculate the center-point of the cluster
          int n_points = 0;
          int x = 0;
          int y = 0;
          for (typename Cluster::iterator s_it = cluster.begin();
               s_it != cluster.end(); ++s_it)
            {
              ScanlineSegment& segment = *s_it;
              for (int i = segment.left; i <= segment.right; i++, n_points++)
                {
                  x += i;
                  y += segment.y;
                }
            }

          x = (int)roundf(x / (float)n_points);
          y = (int)roundf(y / (float)n_points);

          // Reproject and offset point
          float s = (x / half_width) - 1.f;
          float t = -((y / half_height) - 1.f);
          float depth = (float)depth_image[y * width + x];
          joint->x = (tan_half_hfov * depth) * s;
          joint->y = (tan_half_vfov * depth) * t;
          joint->z = depth + params[j].offset;

          // Calculate the confidence of the cluster
          joint->confidence = 0.f;
          for (typename Cluster::iterator s_it = cluster.begin();
               s_it != cluster.end(); ++s_it)
            {
              ScanlineSegment& segment = *s_it;
              int idx = segment.y * width;
              for (int i = segment.left; i <= segment.right; i++)
                {
                  joint->confidence += weights[(idx + i) * n_joints + j];
                }
            }

          // Add the joint to the list
          result->joints[j] = llist_insert_before(result->joints[j],
                                                  llist_new(joint));
        }

      llist_sort(result->joints[j], compare_joints, NULL);
    }

  return result;
}

template InferredJoints*
infer_joints_fast<half>(half*, float*, float*, int32_t, int32_t, uint8_t,
                        JSON_Value*, float, JIParam*);

template InferredJoints*
infer_joints_fast<float>(float*, float*, float*, int32_t, int32_t, uint8_t,
                         JSON_Value*, float, JIParam*);

template<typename FloatT>
InferredJoints*
infer_joints(FloatT* depth_image, float* pr_table, float* weights,
             int32_t width, int32_t height,
             uint8_t n_labels, JSON_Value* joint_map,
             float vfov, JIParam* params)
{
  int n_joints = json_array_get_count(json_array(joint_map));

  JointMapEntry map[n_joints];
  unpack_joint_map(joint_map, map, n_joints);

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

  uint32_t too_many_pixels = (width * height) / 2;

  // Gather pixels above the given threshold
  for (int32_t y = 0, idx = 0; y < height; y++)
    {
      float t = -((y / half_height) - 1.f);
      for (int32_t x = 0; x < width; x++, idx++)
        {
          float s = (x / half_width) - 1.f;
          float depth = (float)depth_image[idx];
          if (!std::isnormal(depth) || depth >= HUGE_DEPTH)
            {
              continue;
            }

          for (uint8_t j = 0; j < n_joints; j++)
            {
              float threshold = params[j].threshold;
              uint32_t joint_idx = j * width * height;

              for (int n = 0; n < map[j].n_labels; n++)
                {
                  uint8_t label = map[j].labels[n];
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

  InferredJoints* result = (InferredJoints*)xmalloc(sizeof(InferredJoints));
  result->n_joints = n_joints;
  result->joints = (LList**)xcalloc(n_joints, sizeof(LList*));

  // Means shift to find joint modes
  for (uint8_t j = 0; j < n_joints; j++)
    {
      if (n_pixels[j] == 0 || n_pixels[j] > too_many_pixels)
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
              // Calculate the confidence of all modes found
              float* last_point = &points[joint_idx * 3];
              Joint* joint = (Joint*)xmalloc(sizeof(Joint));
              joint->x = last_point[0];
              joint->y = last_point[1];
              joint->z = last_point[2] + offset;
              joint->confidence = 0;
              result->joints[j] = llist_new(joint);

              //uint32_t unique_points = 1;

              for (uint32_t p = 0; p < n_pixels[j]; p++)
                {
                  float* point = &points[(joint_idx + p) * 3];
                  if (fabs(point[0]-last_point[0]) >= SHIFT_THRESHOLD ||
                      fabs(point[1]-last_point[1]) >= SHIFT_THRESHOLD ||
                      fabs(point[2]-last_point[2]) >= SHIFT_THRESHOLD)
                    {
                      //unique_points++;
                      last_point = point;
                      joint = (Joint*)xmalloc(sizeof(Joint));
                      joint->x = last_point[0];
                      joint->y = last_point[1];
                      joint->z = last_point[2] + offset;
                      joint->confidence = 0;
                      result->joints[j] = llist_insert_before(result->joints[j],
                                                              llist_new(joint));
                    }
                  joint->confidence += density[joint_idx + p];
                }

              llist_sort(result->joints[j], compare_joints, NULL);

              break;
            }
        }
    }

  xfree(density);
  xfree(points);
  xfree(n_pixels);

  return result;
}

template InferredJoints*
infer_joints<half>(half*, float*, float*, int32_t, int32_t, uint8_t,
                   JSON_Value*, float, JIParam*);

template InferredJoints*
infer_joints<float>(float*, float*, float*, int32_t, int32_t, uint8_t,
                    JSON_Value*, float, JIParam*);

void
free_joints(InferredJoints* joints)
{
  for (int i = 0; i < joints->n_joints; i++)
    {
      llist_free(joints->joints[i], llist_free_cb, NULL);
    }
  xfree(joints->joints);
  xfree(joints);
}

template<typename FloatT>
float*
reproject(FloatT* depth_image, int32_t width, int32_t height,
          float vfov, float threshold, uint32_t* n_points, float* out_cloud)
{
  float half_width = width / 2.f;
  float half_height = height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = vfov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;

  float* point_cloud = out_cloud ? out_cloud :
    (float*)xmalloc(width * height * 3 * sizeof(float));

  *n_points = 0;
  int32_t ty = -1;
  for (int32_t y = 0, idx = 0; y < height; y++)
    {
      float t;
      for (int32_t x = 0; x < width; x++, idx++)
        {
          float depth = (float)depth_image[idx];
          if (!std::isnormal(depth) || depth > threshold)
            {
              continue;
            }

          float s = (x / half_width) - 1.f;
          if (ty != y)
            {
              t = -((y / half_height) - 1.f);
              ty = y;
            }
          uint32_t cloud_idx = (*n_points) * 3;

          point_cloud[cloud_idx] = (tan_half_hfov * depth) * s;
          point_cloud[cloud_idx + 1] = (tan_half_vfov * depth) * t;
          point_cloud[cloud_idx + 2] = depth;

          (*n_points)++;
        }
    }

  if (!out_cloud)
    {
      point_cloud = (float*)
        xrealloc(point_cloud, (*n_points) * 3 * sizeof(float));
    }

  return point_cloud;
}

template float*
reproject<half>(half*, int32_t, int32_t, float, float, uint32_t*, float*);

template float*
reproject<float>(float*, int32_t, int32_t, float, float, uint32_t*, float*);

template<typename FloatT>
FloatT*
project(float* point_cloud, uint32_t n_points, int32_t width, int32_t height,
        float vfov, float background, FloatT* out_depth)
{
  float half_width = width / 2.f;
  float half_height = height / 2.f;
  float aspect = half_width / half_height;

  float vfov_rad = vfov * M_PI / 180.f;
  float tan_half_vfov = tanf(vfov_rad / 2.f);
  float tan_half_hfov = tan_half_vfov * aspect;

  FloatT* depth_image = out_depth ? out_depth :
    (FloatT*)xmalloc(width * height * sizeof(FloatT));
  FloatT bg_half = (FloatT)background;
  for (int32_t i = 0; i < width * height; i++)
    {
      depth_image[i] = bg_half;
    }

  for (uint32_t i = 0, idx = 0; i < n_points; i++, idx += 3)
    {
      float* point = &point_cloud[idx];

      float x = point[0] / (tan_half_hfov * point[2]);
      if (x < -1.0f || x >= 1.0f)
        {
          continue;
        }

      float y = -point[1] / (tan_half_vfov * point[2]);
      if (y < -1.0f || y >= 1.0f)
        {
          continue;
        }

      x = (x + 1.0f) * half_width;
      y = (y + 1.0f) * half_height;

      int32_t col = x;
      int32_t row = y;

      depth_image[row * width + col] = (FloatT)point[2];
    }

  return depth_image;
}

template half*
project<half>(float*, uint32_t, int32_t, int32_t, float, float, half*);

template float*
project<float>(float*, uint32_t, int32_t, int32_t, float, float, float*);
