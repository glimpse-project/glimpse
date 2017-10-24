%module glimpse

%{
#define SWIG_FILE_WITH_INIT
#include "glimpse_python.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%typemap(in) (char** IN_ARRAY1, int DIM1) {
  /* Check if is a list */
  if (PyList_Check($input))
    {
      $2 = PyList_Size($input);
      $1 = (char **) malloc($2*sizeof(char *));
      for (Py_ssize_t i = 0; i < $2; i++)
        {
          PyObject *o = PyList_GetItem($input, i);
          if (PyUnicode_Check(o))
            {
              $1[i] = PyUnicode_AsUTF8(PyList_GetItem($input,i));
            }
          else
            {
              PyErr_Format(PyExc_TypeError, "list must contain strings. %d/%d element was not string.", i, $2);
              free($1);
              return NULL;
            }
        }
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"not a list");
      return NULL;
    }
}

%typemap(freearg) (char** IN_ARRAY1, int DIM1) {
  free((char *) $1);
}

%apply (char** IN_ARRAY1, int DIM1) {(char** aFiles, unsigned int aNFiles)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* aDepthImage, int aHeight, int aWidth)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* aPointCloud, int aNPoints, int aNDims)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aDepth, int* aOutHeight, int* aOutWidth)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aCloud, int* aOutNPoints, int* aOutNDims)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aJoints, int* aOutNJoints, int* aOutNDims)};
%apply (float** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {(float** aLabelPr, int* aOutHeight, int* aOutWidth, int* aNLabels)};

%include "glimpse_python.h"
