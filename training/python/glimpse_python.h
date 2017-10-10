
#include "glimpse.h"

namespace Glimpse
{
  class DepthImage {
    friend class Forest;

    friend DepthImage* ReadEXR(const char* aFileName);

    private:
      half_float::half* mDepthImage;
      uint32_t          mWidth;
      uint32_t          mHeight;

      DepthImage(half_float::half* aDepthImage,
                 uint32_t          aWidth,
                 uint32_t          aHeight);

    public:
      ~DepthImage();
  };

  class Forest {
    private:
      RDTree**     mForest;
      unsigned int mNTrees;

    public:
      Forest(char**       aFiles,
             unsigned int aNFiles);
      ~Forest();

      void inferLabels(DepthImage* aDepthImage,
                       float**     aLabelPr,
                       int*        aOutWidth,
                       int*        aOutHeight,
                       int*        aNLabels);
  };

  DepthImage* ReadEXR(const char* aFileName);
}
