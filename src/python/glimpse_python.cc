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

#include "glimpse_python.h"
#include "xalloc.h"

template<typename FloatT>
static FloatT*
project(float* point_cloud, int n_points, int width, int height,
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
    for (int i = 0; i < width * height; i++)
    {
        depth_image[i] = bg_half;
    }

    for (int i = 0, idx = 0; i < n_points; i++, idx += 3)
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

        int col = x;
        int row = y;

        depth_image[row * width + col] = (FloatT)point[2];
    }

    return depth_image;
}

template half*
project<half>(float*, int, int, int, float, float, half*);

template float*
project<float>(float*, int, int, int, float, float, float*);

template<typename FloatT>
float*
reproject(FloatT* depth_image, int width, int height,
          float vfov, float threshold, int* n_points, float* out_cloud)
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
    int ty = -1;
    for (int y = 0, idx = 0; y < height; y++)
    {
        float t;
        for (int x = 0; x < width; x++, idx++)
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
            int cloud_idx = (*n_points) * 3;

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
reproject<half>(half*, int, int, float, float, int*, float*);

template float*
reproject<float>(float*, int, int, float, float, int*, float*);


namespace Glimpse {

using half_float::half;

DepthImage::DepthImage(half* aDepthImage, uint32_t aWidth, uint32_t aHeight)
{
  mDepthImage = aDepthImage;
  mWidth = aWidth;
  mHeight = aHeight;
  mValid = true;
}

DepthImage::DepthImage(const char* aFileName)
{
  mValid = false;

  mDepthImage = NULL;
  IUImageSpec spec = { 0, 0, IU_FORMAT_HALF };
  if (iu_read_exr_from_file(aFileName, &spec, (void**)&mDepthImage) == SUCCESS)
    {
      mWidth = spec.width;
      mHeight = spec.height;
      mValid = true;
    }
}

DepthImage::DepthImage(float* aDepthImage, int aHeight, int aWidth)
{
  mValid = true;
  mWidth = aWidth;
  mHeight = aHeight;
  mDepthImage = (half*)xmalloc(aWidth * aHeight * sizeof(half));
  for (int i = 0; i < aWidth * aHeight; i++)
    {
      mDepthImage[i] = (half)aDepthImage[i];
    }
}

DepthImage::~DepthImage()
{
  if (mValid)
    {
      xfree(mDepthImage);
      mValid = false;
    }
}

void
DepthImage::writeEXR(const char* aFileName)
{
  IUImageSpec spec = { (int)mWidth, (int)mHeight, IU_FORMAT_HALF };
  if (iu_write_exr_to_file(aFileName, &spec, (void*)mDepthImage,
                           IU_FORMAT_HALF) != SUCCESS)
    {
      fprintf(stderr, "Error writing EXR file '%s'\n", aFileName);
    }
}

void
DepthImage::asArray(float** aDepth, int* aOutHeight, int* aOutWidth)
{
  *aDepth = (float*)xmalloc(mWidth * mHeight * sizeof(float));
  for (uint32_t i = 0; i < mWidth * mHeight; i++)
    {
      (*aDepth)[i] = (float)mDepthImage[i];
    }
  *aOutWidth = mWidth;
  *aOutHeight = mHeight;
}

void
DepthImage::asPointCloud(float aVFOV, float aThreshold, float** aCloud,
                         int* aOutNPoints, int* aOutNDims)
{
  uint32_t n_points = 0;
  *aCloud = reproject(mDepthImage, mWidth, mHeight, aVFOV,
                      aThreshold, &n_points);
  *aOutNPoints = (int)n_points;
  *aOutNDims = 3;

  if (n_points == 0)
    {
      *aCloud = (float*)xmalloc(sizeof(float));
    }
}

DepthImage*
DepthImageFromPointCloud(float* aPointCloud, int aNPoints, int aNDims,
                         int aHeight, int aWidth, float aVFOV,
                         float aBackground)
{
  if (aNDims != 3 || aNPoints < 1 || aWidth < 1 || aHeight < 1 || aVFOV <= 0.f)
    {
      return NULL;
    }

  half* depth_image = project(aPointCloud, aNPoints, aWidth, aHeight, aVFOV,
                              aBackground);
  return new DepthImage(depth_image, aWidth, aHeight);
}

Forest::Forest(const char** aFiles, unsigned int aNFiles)
{
  RDTree** forest = read_json_forest(aFiles, aNFiles);
  mForest = forest;
  mNTrees = aNFiles;
}

Forest::~Forest()
{
  if (mForest)
    {
      free_forest(mForest, mNTrees);
      mForest = NULL;
    }
}

void
Forest::inferLabels(DepthImage* aDepthImage, float** aLabelPr,
                    int* aOutHeight, int* aOutWidth, int* aNLabels)
{
  if (!mForest || !aDepthImage->mValid)
    {
      return;
    }

  *aLabelPr = infer_labels(mForest, mNTrees, aDepthImage->mDepthImage,
                           aDepthImage->mWidth, aDepthImage->mHeight);
  *aOutWidth = aDepthImage->mWidth;
  *aOutHeight = aDepthImage->mHeight;
  *aNLabels = mForest[0]->header.n_labels;
}

JointMap::JointMap(char* aJointMap, char* aJointInferenceParams)
{
  mValid = false;
  mParams = read_jip(aJointInferenceParams);
  if (mParams)
    {
      mJointMap = json_parse_file(aJointMap);
      if (mJointMap)
        {
          mValid = true;
        }
      else
        {
          fprintf(stderr, "Error reading joint map\n");
          free_jip(mParams);
        }
    }
  else
    {
      fprintf(stderr, "Error reading joint parameters\n");
    }
}

JointMap::~JointMap()
{
  if (mValid)
    {
      mValid = false;
      json_value_free(mJointMap);
      free_jip(mParams);
    }
}

void
JointMap::inferJoints(Forest* aForest, DepthImage* aDepthImage,
                      float** aJoints, int* aOutNJoints, int* aOutNDims)
{
  if (!mValid || !aForest->mForest || !aDepthImage->mValid)
    {
      return;
    }

  float* pr_table;
  int width, height, n_labels;
  aForest->inferLabels(aDepthImage, &pr_table, &width, &height, &n_labels);

  float* weights = calc_pixel_weights(aDepthImage->mDepthImage,
                                      pr_table, width, height, n_labels,
                                      mJointMap);

  InferredJoints* result =
    infer_joints(aDepthImage->mDepthImage, pr_table, weights,
                 aDepthImage->mWidth, aDepthImage->mHeight,
                 aForest->mForest[0]->header.n_labels,
                 mJointMap,
                 aForest->mForest[0]->header.fov,
                 mParams->joint_params);

  xfree(weights);
  xfree(pr_table);

  // TODO: Create an object equivalent of InferredJoints for bindings
  *aJoints = (float*)xcalloc(result->n_joints, sizeof(float) * 3);
  for (int i = 0; i < result->n_joints; i++)
    {
      if (!result->joints[i])
        {
          continue;
        }

      (*aJoints)[i * 3] = ((Joint*)result->joints[i]->data)->x;
      (*aJoints)[i * 3 + 1] = ((Joint*)result->joints[i]->data)->y;
      (*aJoints)[i * 3 + 2] = ((Joint*)result->joints[i]->data)->z;
    }

  free_joints(result);

  *aOutNJoints = mParams->header.n_joints;
  *aOutNDims = 3;
}

}

