
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <png.h>
#include "tinyexr.h"
#include "half.hpp"

#include "train_utils.h"
#include "image_utils.h"
#include "xalloc.h"
#include "llist.h"

using half_float::half;

typedef struct {
  uint32_t n_images;      // Number of training images
  uint8_t  n_joints;      // Number of joints
  uint32_t limit;         // Limit to number of training images
  uint32_t skip;          // Number of images to skip
  bool     shuffle;       // Whether to shuffle images
  LList*   paths;         // List of label, depth and joint file paths
  IUImageSpec spec;       // Image specification
  half*    depth_images;  // Depth image data
  uint8_t* label_images;  // Label image data
  float*   joint_data;    // Joint data
  bool     gather_depth;  // Whether to load depth images
  bool     gather_label;  // Whether to load label images
  bool     gather_joints; // Whether to gather joint data
} TrainData;

static bool
gather_cb(const char* label_path, const char* depth_path,
          const char* joint_path, TrainData* data)
{
  if (data->n_images >= data->skip)
    {
      char** paths = (char**)xmalloc(3 * sizeof(char*));
      paths[0] = label_path ? strdup(label_path) : NULL;
      paths[1] = depth_path ? strdup(depth_path) : NULL;
      paths[2] = joint_path ? strdup(joint_path) : NULL;
      data->paths = llist_insert_after(data->paths, llist_new(paths));
    }
  ++data->n_images;
  return data->shuffle || (data->n_images < data->limit + data->skip);
}

static bool
gather_train_files(const char* label_dir_path,
                   const char* depth_dir_path,
                   const char* joint_dir_path,
                   TrainData*  data)
{
  struct stat st;
  DIR *dir;
  struct dirent *entry;
  const char* dir_path;
  char *ext;
  bool cont;

  if (!label_dir_path && !depth_dir_path && !joint_dir_path)
    {
      return true;
    }

  cont = true;
  dir_path = label_dir_path ? label_dir_path :
                              (depth_dir_path ? depth_dir_path :
                                                joint_dir_path);
  dir = opendir(dir_path);

  while (cont && (entry = readdir(dir)) != NULL)
    {
      char *next_label_path = NULL;
      char *next_depth_path = NULL;
      char *next_joint_path = NULL;

      if (strcmp(entry->d_name, ".") == 0 ||
          strcmp(entry->d_name, "..") == 0)
          continue;

      if ((label_dir_path &&
           asprintf(&next_label_path, "%s/%s",
                    label_dir_path, entry->d_name) == -1) ||
          (depth_dir_path &&
           asprintf(&next_depth_path, "%s/%s",
                    depth_dir_path, entry->d_name) == -1) ||
          (joint_dir_path &&
           asprintf(&next_joint_path, "%s/%s",
                    joint_dir_path, entry->d_name) == -1))
        {
          fprintf(stderr, "Error creating file paths\n");
          exit(1);
        }

      stat(label_dir_path ? next_label_path :
                            (depth_dir_path ? next_depth_path :
                                              next_joint_path), &st);
      if (S_ISDIR(st.st_mode))
        {
          cont = gather_train_files(next_label_path, next_depth_path,
                                    next_joint_path, data);
        }
      else if ((dir_path == label_dir_path &&
                (ext = strstr(entry->d_name, ".png")) && ext[4] == '\0') ||
               (dir_path == depth_dir_path &&
                (ext = strstr(entry->d_name, ".exr")) && ext[4] == '\0') ||
               (dir_path == joint_dir_path &&
                (ext = strstr(entry->d_name, ".jnt")) && ext[4] == '\0'))
        {
          if (label_dir_path)
            {
              if (depth_dir_path)
                {
                  strcpy(next_depth_path + strlen(next_depth_path) - 4, ".exr");
                }
              if (joint_dir_path)
                {
                  strcpy(next_joint_path + strlen(next_joint_path) - 4, ".jnt");
                }
            }
          else if (depth_dir_path && joint_dir_path)
            {
              strcpy(next_joint_path + strlen(next_joint_path) - 4, ".jnt");
            }

          cont =
            gather_cb(next_label_path, next_depth_path, next_joint_path, data);
        }

      if (label_dir_path)
        {
          free(next_label_path);
        }
      if (depth_dir_path)
        {
          free(next_depth_path);
        }
      if (joint_dir_path)
        {
          free(next_joint_path);
        }
    }

  closedir(dir);

  return cont;
}

static bool
free_train_data_cb(LList* node, uint32_t index, void* userdata)
{
  char** paths = (char**)node->data;
  if (paths[0])
    {
      xfree(paths[0]);
    }
  if (paths[1])
    {
      xfree(paths[1]);
    }
  if (paths[2])
    {
      xfree(paths[2]);
    }
  xfree(paths);
  return true;
}

static void
verify_metadata(TrainData* data, char* filename,
                int32_t width, int32_t height, uint8_t n_joints)
{
  if (width && height && (data->spec.width != width ||
                          data->spec.height != height))
    {
      if (data->spec.width == 0 && data->spec.height == 0)
        {
          data->spec.width = width;
          data->spec.height = height;
          size_t n_pixels = width * height * data->n_images;
          if (data->gather_label)
            {
              data->label_images = (uint8_t*)
                xmalloc(n_pixels * sizeof(uint8_t));
            }
          if (data->gather_depth)
            {
              data->depth_images = (half*)
                xmalloc(n_pixels * sizeof(half));
            }
        }
      else
        {
          fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
                  filename, width, height, data->spec.width, data->spec.height);
          exit(1);
        }
    }

  if (n_joints && data->n_joints != n_joints)
    {
      if (data->n_joints == 0)
        {
          data->n_joints = n_joints;
          if (data->gather_joints)
            {
              data->joint_data = (float*)
                xmalloc(data->n_images * n_joints * 3 * sizeof(float));
            }
        }
      else
        {
          fprintf(stderr, "%s: joint number mismatch (%u), expected (%u)\n",
                  filename, (uint32_t)n_joints, (uint32_t)data->n_joints);
          exit(1);
        }
    }
}

static bool
train_data_cb(LList* node, uint32_t index, void* userdata)
{
  FILE* fp;
  int width, height;

  char** paths = (char**)node->data;
  char* label_path = paths[0];
  char* depth_path = paths[1];
  char* joint_path = paths[2];
  TrainData *data = (TrainData*)userdata;

  if (label_path && data->gather_label)
    {
      if (index == 0)
        {
          if (iu_verify_png_from_file(label_path, &data->spec) != SUCCESS)
            {
              fprintf(stderr, "Failed to verify image '%s'\n", label_path);
              exit(1);
            }
          size_t n_pixels =
            data->spec.width * data->spec.height * data->n_images;
          data->label_images = (uint8_t*)xmalloc(n_pixels * sizeof(uint8_t));
          if (data->gather_depth)
            {
              data->depth_images = (half*)xmalloc(n_pixels * sizeof(half));
            }
        }

      char* output = (char*)
        &data->label_images[index * data->spec.width * data->spec.height];
      if (iu_read_png_from_file(label_path, &data->spec, &output) != SUCCESS)
        {
          fprintf(stderr, "Failed to read image '%s'\n", label_path);
          exit(1);
        }
    }

  if (depth_path && data->gather_depth)
    {
      // Read depth file
      EXRVersion version;
      EXRHeader header;
      int ret;
      const char *err = NULL;

      ret = ParseEXRVersionFromFile(&version, depth_path);

      if (ret != TINYEXR_SUCCESS)
        {
          fprintf(stderr, "Error %02d reading EXR version\n", ret);
          exit(1);
        }

      if (version.multipart || version.non_image)
        {
          fprintf(stderr, "Can't load multipart or DeepImage EXR image\n");
          exit(1);
        }

      ret = ParseEXRHeaderFromFile(&header, &version, depth_path, &err);

      if (ret != TINYEXR_SUCCESS)
        {
          fprintf(stderr, "Error %02d reading EXR header: %s\n", ret, err);
          exit(1);
        }

      if (header.num_channels != 1 ||
          header.channels[0].pixel_type != TINYEXR_PIXELTYPE_HALF)
        {
          fprintf(stderr, "Expected single-channel half-float EXR\n");
          exit(1);
        }

      width = header.data_window[2] - header.data_window[0] + 1;
      height = header.data_window[3] - header.data_window[1] + 1;
      verify_metadata(data, depth_path, width, height, 0);

      EXRImage exr_image;
      InitEXRImage(&exr_image);

      ret = LoadEXRImageFromFile(&exr_image, &header, depth_path, &err);

      if (ret != TINYEXR_SUCCESS)
        {
          fprintf(stderr, "Error %02d reading EXT file: %s\n", ret, err);
          exit(1);
        }

      memcpy(&data->depth_images[index * width * height],
             &exr_image.images[0][0],
             width * height * sizeof(half));

      FreeEXRImage(&exr_image);
    }

  // Read joint data
  if (joint_path && data->gather_joints)
    {
      if (!(fp = fopen(joint_path, "rb")))
        {
          fprintf(stderr, "Error opening joint file '%s'\n", joint_path);
          exit(1);
        }

      if (fseek(fp, 0, SEEK_END) == -1)
        {
          fprintf(stderr, "Error seeking to end of joint file '%s'\n",
                  joint_path);
          exit(1);
        }

      long n_bytes = ftell(fp);
      if (n_bytes % sizeof(float) != 0 ||
          (n_bytes % sizeof(float)) % 3 != 0)
        {
          fprintf(stderr, "Unexpected joint file size in '%s'\n",
                  joint_path);
          exit(1);
        }

      uint8_t n_joints = (uint8_t)((n_bytes / sizeof(float)) / 3);
      verify_metadata(data, joint_path, 0, 0, n_joints);

      if (fseek(fp, 0, SEEK_SET) == -1)
        {
          fprintf(stderr, "Error seeking to start of joint file '%s'\n",
                  joint_path);
          exit(1);
        }

      float* joints = &data->joint_data[index * n_joints * 3];
      if (fread(joints, sizeof(float) * 3, n_joints, fp) != n_joints)
        {
          fprintf(stderr, "%s: Error reading joints\n", joint_path);
          exit(1);
        }

      if (fclose(fp) != 0)
        {
          fprintf(stderr, "Error closing joint file '%s'\n", joint_path);
          exit(1);
        }
    }

  // Free file path copies
  free_train_data_cb(node, index, userdata);

  return true;
}

void
gather_train_data(const char* label_dir_path, const char* depth_dir_path,
                  const char* joint_dir_path,
                  uint32_t limit, uint32_t skip, bool shuffle,
                  uint32_t* out_n_images, uint8_t* out_n_joints,
                  int32_t* out_width, int32_t* out_height,
                  half** out_depth_images, uint8_t** out_label_images,
                  float** out_joints)
{
  TrainData data = {
    0,                                    // Number of training images
    0,                                    // Number of joints
    limit,                                // Limit to number of training images
    skip,                                 // Number of images to skip
    shuffle,                              // Whether to shuffle images
    NULL,                                 // Image paths
    {0,0,8,1},                            // Image specification
    NULL,                                 // Depth image data
    NULL,                                 // Label image data
    NULL,                                 // Joint data
    (depth_dir_path && out_depth_images), // Whether to gather depth images
    (label_dir_path && out_label_images), // Whether to gather label images
    (joint_dir_path && out_joints)        // Whether to gather joint data
  };

  gather_train_files(label_dir_path, depth_dir_path, joint_dir_path, &data);
  data.n_images = (data.n_images > data.skip) ? data.n_images - data.skip : 0;

  *out_n_images = (data.n_images < data.limit) ? data.n_images : data.limit;
  printf("Processing %d training images...\n", *out_n_images);

  data.paths = llist_first(data.paths);
  if (data.shuffle)
    {
      data.paths = llist_slice(llist_shuffle(data.paths),
                               0, data.limit, free_train_data_cb, NULL);
    }
  llist_foreach(data.paths, train_data_cb, (void*)&data);
  llist_free(data.paths, NULL, NULL);

  if (out_width)
    {
      *out_width = data.spec.width;
    }
  if (out_height)
    {
      *out_height = data.spec.height;
    }
  if (out_n_joints)
    {
      *out_n_joints = data.n_joints;
    }
  if (out_depth_images)
    {
      *out_depth_images = data.depth_images;
    }
  if (out_label_images)
    {
      *out_label_images = data.label_images;
    }
  if (out_joints)
    {
      *out_joints = data.joint_data;
    }
}

