
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "half.hpp"
#include "train_utils.h"
#include "image_utils.h"
#include "xalloc.h"
#include "llist.h"
#include "parson.h"

using half_float::half;

#define xsnprintf(dest, fmt, ...) do { \
        if (snprintf(dest, sizeof(dest), fmt,  __VA_ARGS__) >= (int)sizeof(dest)) \
            exit(1); \
    } while(0)

typedef struct {
  double   vertical_fov;  // Field of view used to render depth images
  uint32_t n_images;      // Number of training images
  uint8_t  n_joints;      // Number of joints
  uint32_t limit;         // Limit to number of training images
  uint32_t skip;          // Number of images to skip
  bool     shuffle;       // Whether to shuffle images
  LList*   paths;         // List of label, depth and joint file paths
  IUImageSpec label_spec; // Label image specification
  IUImageSpec depth_spec; // Depth image specification
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
gather_train_files(const char* top_src_dir,
                   const char* index_name,
                   TrainData* data)
{
  char index_filename[1024];
  bool cont = true;

  xsnprintf(index_filename, "%s/index.%s", top_src_dir, index_name);

  FILE* fp = fopen(index_filename, "r");
  if (!fp)
    {
      fprintf(stderr, "Failed to open index %s\n", index_filename);
      exit(1);
    }

  char* line = NULL;
  size_t line_buf_len = 0;
  int line_len;
  while (cont && (line_len = getline(&line, &line_buf_len, fp)) != -1)
    {
      char next_labels_path[1024];
      char next_depth_path[1024];
      char next_jnt_path[1024];

      if (line_len <= 1)
        continue;

      /* remove the trailing newline from the line */
      line[line_len - 1] = '\0';

      xsnprintf(next_labels_path, "%s/labels/%s.png", top_src_dir, line);
      xsnprintf(next_depth_path, "%s/depth/%s.exr", top_src_dir, line);
      xsnprintf(next_jnt_path, "%s/labels/%s.jnt", top_src_dir, line);

      cont = gather_cb(next_labels_path, next_depth_path, next_jnt_path, data);
    }

  free(line);

  fclose(fp);

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
validate_storage(TrainData* data, const char* filename, uint8_t n_joints)
{
  if ((!data->label_images || !data->depth_images) &&
      data->gather_label && data->gather_depth &&
      ((data->label_spec.width && data->depth_spec.width &&
        data->label_spec.width != data->depth_spec.width) ||
       (data->label_spec.height && data->depth_spec.height &&
        data->label_spec.height != data->depth_spec.height)))
    {
      fprintf(stderr,
              "%s: Label/Depth image size mismatch (%dx%d) != (%dx%d)\n",
              filename, data->label_spec.width, data->label_spec.height,
              data->depth_spec.width, data->depth_spec.height);
    }

  if (!data->label_images && data->gather_label &&
      data->label_spec.width && data->label_spec.height)
    {
      size_t n_pixels = data->label_spec.width * data->label_spec.height *
                        data->n_images;
      data->label_images = (uint8_t*)
        xmalloc(n_pixels * sizeof(uint8_t));
    }
  if (!data->depth_images && data->gather_depth &&
      data->depth_spec.width && data->depth_spec.height)
    {
      size_t n_pixels = data->depth_spec.width * data->depth_spec.height *
                        data->n_images;
      data->depth_images = (half*)
        xmalloc(n_pixels * sizeof(half));
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
  char** paths = (char**)node->data;
  char* label_path = paths[0];
  char* depth_path = paths[1];
  char* joint_path = paths[2];
  TrainData *data = (TrainData*)userdata;

  // Read label image
  if (label_path && data->gather_label)
    {
      uint8_t* output = &data->label_images[
        index * data->label_spec.width * data->label_spec.height];
      if (iu_read_png_from_file(label_path, &data->label_spec, &output,
                                NULL, // palette output
                                NULL) // palette size
          != SUCCESS)
        {
          fprintf(stderr, "Failed to read image '%s'\n", label_path);
          exit(1);
        }
    }

  // Read depth image
  if (depth_path && data->gather_depth)
    {
      void* output = &data->depth_images[
        index * data->depth_spec.width * data->depth_spec.height];
      if (iu_read_exr_from_file(depth_path, &data->depth_spec, &output) !=
          SUCCESS)
        {
          fprintf(stderr, "Failed to read image '%s'\n", depth_path);
          exit(1);
        }
    }

  // Read joint data
  if (joint_path && data->gather_joints)
    {
      FILE* fp;
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
      validate_storage(data, joint_path, n_joints);

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
gather_train_data(const char* data_dir,
                  const char* index_name,
                  uint32_t limit, uint32_t skip, bool shuffle,
                  uint32_t* out_n_images, uint8_t* out_n_joints,
                  int32_t* out_width, int32_t* out_height,
                  half** out_depth_images, uint8_t** out_label_images,
                  float** out_joints,
                  uint8_t* out_n_labels,
                  float* out_fov)
{
  char meta_filename[1024];

  TrainData data = {
    0,                                    // Field of view used to render depth
    0,                                    // Number of training images
    0,                                    // Number of joints
    limit,                                // Limit to number of training images
    skip,                                 // Number of images to skip
    shuffle,                              // Whether to shuffle images
    NULL,                                 // Image paths
    {0,0,IU_FORMAT_U8},                   // Label image specification
    {0,0,IU_FORMAT_HALF},                 // Depth image specification
    NULL,                                 // Depth image data
    NULL,                                 // Label image data
    NULL,                                 // Joint data
    !!out_depth_images,                   // Whether to gather depth images
    !!out_label_images,                   // Whether to gather label images
    !!out_joints                          // Whether to gather joint data
  };

  xsnprintf(meta_filename, "%s/meta.json", data_dir);

  JSON_Value* meta = json_parse_file(meta_filename);

  int n_labels = json_object_get_number(json_object(meta), "n_labels");
  JSON_Object* camera = json_object_get_object(json_object(meta), "camera");

  data.vertical_fov = json_object_get_number(camera, "vertical_fov");
  int width = json_object_get_number(camera, "width");
  int height = json_object_get_number(camera, "height");

  json_value_free(meta);

  data.label_spec.width = width;
  data.label_spec.height = height;
  data.depth_spec.width = width;
  data.depth_spec.height = height;

  gather_train_files(data_dir, index_name, &data);

  data.n_images = (data.n_images > data.skip) ? data.n_images - data.skip : 0;

  /* This will allocate memory for all our image data based on the camera size
   * read from meta.json, and the number of frames found in the index loaded by
   * gather_train_failes...
   */
  validate_storage(&data, "meta.json", 0);

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
      *out_width = data.gather_label ?
        data.label_spec.width : data.depth_spec.width;
    }
  if (out_height)
    {
      *out_height = data.gather_label ?
        data.label_spec.height : data.depth_spec.height;
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
  if (out_n_labels)
    {
      *out_n_labels = n_labels;
    }
  if (out_fov)
    {
      *out_fov = data.vertical_fov;
    }
}

