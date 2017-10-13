
#include "glimpse_python.h"
#include "xalloc.h"

namespace Glimpse {

using half_float::half;

DepthImage::DepthImage(half* aDepthImage, uint32_t aWidth, uint32_t aHeight)
{
  mDepthImage = aDepthImage;
  mWidth = aWidth;
  mHeight = aHeight;
}

DepthImage::~DepthImage()
{
  if (mDepthImage)
    {
      xfree(mDepthImage);
      mDepthImage = NULL;
    }
}

DepthImage*
ReadEXR(const char* aFileName)
{
  half* output = NULL;
  IUImageSpec spec = { 0, 0, IU_FORMAT_HALF };
  if (iu_read_exr_from_file(aFileName, &spec, (void**)&output) == SUCCESS)
    {
      return new DepthImage(output, spec.width, spec.height);
    }

  return NULL;
}

Forest::Forest(char** aFiles, unsigned int aNFiles)
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
  if (!mForest)
    {
      return;
    }

  *aLabelPr = infer_labels(mForest, mNTrees, aDepthImage->mDepthImage,
                           aDepthImage->mWidth, aDepthImage->mHeight);
  *aOutWidth = aDepthImage->mWidth;
  *aOutHeight = aDepthImage->mHeight;
  *aNLabels = mForest[0]->header.n_labels;
}

}

