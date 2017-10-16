
#include "glimpse.h"

namespace Glimpse
{
  class DepthImage {
    friend class Forest;
    friend class JointMap;

    private:
      bool              mValid;
      half_float::half* mDepthImage;
      uint32_t          mWidth;
      uint32_t          mHeight;

      DepthImage(half_float::half* aDepthImage,
                 uint32_t          aWidth,
                 uint32_t          aHeight);

    public:
      DepthImage(const char* aFileName);
      ~DepthImage();
  };

  class Forest {
    friend class JointMap;

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

  class JointMap {
    private:
      bool mValid;
      JIParams* mParams;
      LList** mJointMap;
      char**  mJointNames;

    public:
      JointMap(char* aJointMap, char* aJointInferenceParams);
      ~JointMap();

      void inferJoints(Forest*     aForest,
                       DepthImage* aDepthImage,
                       float**     aJoints,
                       int*        aOutNJoints,
                       int*        aOutNDims);
  };
}
