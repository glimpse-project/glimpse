
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <png.h>

#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImathBox.h>

#include "train_utils.h"
#include "xalloc.h"
#include "llist.h"

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

typedef struct {
  uint32_t n_images;      // Number of training images
  uint8_t  n_joints;      // Number of joints
  uint32_t limit;         // Limit to number of training images
  uint32_t skip;          // Number of images to skip
  bool     shuffle;       // Whether to shuffle images
  LList*   paths;   // List of label, depth and joint file paths,
  int32_t  width;         // Image width
  int32_t  height;        // Image height
  float*   depth_images;  // Depth image data
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

      stat(next_label_path, &st);
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
  if (width && height && (data->width != width || data->height != height))
    {
      if (data->width == 0 && data->height == 0)
        {
          data->width = width;
          data->height = height;
          size_t n_pixels = width * height * data->n_images;
          if (data->gather_label)
            {
              data->label_images = (uint8_t*)
                xmalloc(n_pixels * sizeof(uint8_t));
            }
          if (data->gather_depth)
            {
              data->depth_images = (float*)
                xmalloc(n_pixels * sizeof(float));
            }
        }
      else
        {
          fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
                  filename, width, height, data->width, data->height);
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
                xmalloc(n_joints * data->n_images * sizeof(float) * 3);
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

  if (label_path)
    {
      unsigned char header[8]; // 8 is the maximum size that can be checked
      png_structp png_ptr;
      png_infop info_ptr;

      // Open label file
      if (!(fp = fopen(label_path, "rb")))
        {
          fprintf(stderr, "Failed to open image '%s'\n", label_path);
          exit(1);
        }

      // Load label file
      if (fread(header, 1, 8, fp) != 8)
        {
          fprintf(stderr, "Error reading header of %s\n", label_path);
        }
      if (png_sig_cmp(header, 0, 8))
        {
          fprintf(stderr, "%s was not recognised as a PNG file\n", label_path);
          exit(1);
        }

      // Start reading label png
      png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!png_ptr)
        {
          fprintf(stderr, "png_create_read_struct failed\n");
          exit(1);
        }

      info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr)
        {
          fprintf(stderr, "png_create_info_struct failed\n");
          exit(1);
        }

      if (setjmp(png_jmpbuf(png_ptr)))
        {
          fprintf(stderr, "libpng setjmp failure\n");
          exit(1);
        }

      png_init_io(png_ptr, fp);
      png_set_sig_bytes(png_ptr, 8);

      png_read_info(png_ptr, info_ptr);

      width = png_get_image_width(png_ptr, info_ptr);
      height = png_get_image_height(png_ptr, info_ptr);

      verify_metadata(data, label_path, width, height, 0);

      // Verify pixel type
      if (png_get_color_type(png_ptr, info_ptr) != 0)
        {
          fprintf(stderr, "%s: Expected a grayscale PNG\n", label_path);
          exit(1);
        }

      if (png_get_bit_depth(png_ptr, info_ptr) != 8)
        {
          fprintf(stderr, "%s: Expected 8-bit grayscale\n", label_path);
          exit(1);
        }

      // Start reading data
      if (data->gather_label)
        {
          int row_stride = png_get_rowbytes(png_ptr, info_ptr);
          png_bytep* input_rows = (png_bytep *)
            xmalloc(sizeof(png_bytep*) * height);
          png_bytep input_data = (png_bytep)
            xmalloc(row_stride * height * sizeof(png_bytep));

          for (int y = 0; y < height; y++)
            {
              input_rows[y] = (png_byte*)input_data + row_stride * y;
            }

          if (setjmp(png_jmpbuf(png_ptr)))
            {
              fprintf(stderr, "%s: png_read_image_error\n", label_path);
              exit(1);
            }

          png_read_image(png_ptr, input_rows);

          // Copy label image data into training context struct
          for (int y = 0, src_idx = 0, dest_idx = index * width * height;
               y < height;
               y++, src_idx += row_stride, dest_idx += width)
            {
              memcpy(&data->label_images[dest_idx], &input_data[src_idx], width);
            }

          xfree(input_rows);
          xfree(input_data);
        }

      // Close the label file
      if (fclose(fp) != 0)
        {
          fprintf(stderr, "Error closing label file '%s'\n", label_path);
          exit(1);
        }

      // Free data associated with PNG reading
      png_destroy_info_struct(png_ptr, &info_ptr);
      png_destroy_read_struct(&png_ptr, NULL, NULL);
    }

  if (depth_path)
    {
      // Read depth file
      float* dest;
      InputFile in_file(depth_path);
      Box2i dw = in_file.header().dataWindow();

      width = dw.max.x - dw.min.x + 1;
      height = dw.max.y - dw.min.y + 1;

      verify_metadata(data, depth_path, width, height, 0);

      const ChannelList& channels = in_file.header().channels();
      if (!channels.findChannel("Y"))
        {
          fprintf(stderr, "%s: Only expected to load greyscale EXR files "
                  "with a single 'Y' channel\n", depth_path);
          exit(1);
        }

      if (data->gather_depth)
        {
          dest = &data->depth_images[index * width * height];

          FrameBuffer frameBuffer;
          frameBuffer.insert("Y",
                             Slice (FLOAT,
                                    (char *)dest,
                                    sizeof(float),           // x stride,
                                    sizeof(float) * width)); // y stride

          in_file.setFrameBuffer(frameBuffer);
          in_file.readPixels(dw.min.y, dw.max.y);
        }
    }

  // Read joint data
  if (joint_path)
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

      float* joints = &data->joint_data[index * sizeof(float) * 3 * n_joints];
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
                  float** out_depth_images, uint8_t** out_label_images,
                  float** out_joints)
{
  TrainData data = {
    0,                        // Number of training images
    0,                        // Number of joints
    limit,                    // Limit to number of training images
    skip,                     // Number of images to skip
    shuffle,                  // Whether to shuffle images
    NULL,                     // Image paths
    0,                        // Image width
    0,                        // Image height
    NULL,                     // Depth image data
    NULL,                     // Label image data
    NULL,                     // Joint data
    out_depth_images != NULL, // Whether to gather depth images
    out_label_images != NULL, // Whether to gather label images
    out_joints != NULL        // Whether to gather joint data
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
      *out_width = data.width;
    }
  if (out_height)
    {
      *out_height = data.height;
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

