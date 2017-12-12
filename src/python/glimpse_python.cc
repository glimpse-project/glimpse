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

  *aJoints = infer_joints(aDepthImage->mDepthImage, pr_table, weights,
                          aDepthImage->mWidth, aDepthImage->mHeight,
                          aForest->mForest[0]->header.n_labels,
                          mJointMap,
                          aForest->mForest[0]->header.fov,
                          mParams->joint_params);

  xfree(weights);
  xfree(pr_table);

  *aOutNJoints = mParams->header.n_joints;
  *aOutNDims = 3;
}

}

