
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

DepthImage::~DepthImage()
{
  if (mValid)
    {
      xfree(mDepthImage);
      mValid = false;
    }
}

Forest::Forest(const char** aFiles, unsigned int aNFiles)
{
  RDTree** forest = read_forest(aFiles, aNFiles);
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
                    int* aOutWidth, int* aOutHeight, int* aNLabels)
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
          free_jip(mParams);
        }
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

