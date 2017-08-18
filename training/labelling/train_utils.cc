
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
  uint32_t n_images;     // Number of training images
  uint32_t limit;        // Limit to number of training images
  bool     shuffle;      // Whether to shuffle images
  LList*   image_paths;  // List of label and depth file paths,
  int32_t  width;        // Image width
  int32_t  height;       // Image height
  float*   depth_images; // Depth image data
  uint8_t* label_images; // Label image data
} TrainData;

static bool
gather_cb(const char* label_path, const char* depth_path, TrainData* data)
{
  char** paths = (char**)xmalloc(2 * sizeof(char*));
  paths[0] = strdup(label_path);
  paths[1] = strdup(depth_path);
  data->image_paths = llist_insert_after(data->image_paths, llist_new(paths));
  ++data->n_images;
  return data->shuffle || (data->n_images < data->limit);
}

static bool
gather_train_files(const char* label_dir_path,
                   const char* depth_dir_path,
                   TrainData*  data)
{
  struct stat st;
  DIR *label_dir;
  struct dirent *label_entry;
  char *ext;
  bool cont;

  cont = true;
  label_dir = opendir(label_dir_path);

  while (cont && (label_entry = readdir(label_dir)) != NULL)
    {
      char *next_label_path, *next_depth_path;

      if (strcmp(label_entry->d_name, ".") == 0 ||
          strcmp(label_entry->d_name, "..") == 0)
          continue;

      if (asprintf(&next_label_path, "%s/%s",
                   label_dir_path, label_entry->d_name) == -1 ||
          asprintf(&next_depth_path, "%s/%s",
                   depth_dir_path, label_entry->d_name) == -1)
        {
          fprintf(stderr, "Error creating file paths\n");
          exit(1);
        }

      stat(next_label_path, &st);
      if (S_ISDIR(st.st_mode))
        {
          cont = gather_train_files(next_label_path, next_depth_path, data);
        }
      else if ((ext = strstr(label_entry->d_name, ".png")) && ext[4] == '\0')
        {
          strcpy(next_depth_path + strlen(next_depth_path) - 4, ".exr");
          cont = gather_cb(next_label_path, next_depth_path, data);
        }

      free(next_label_path);
      free(next_depth_path);
    }

  closedir(label_dir);

  return cont;
}

static bool
free_train_data_cb(LList* node, uint32_t index, void* userdata)
{
  char** paths = (char**)node->data;
  xfree(paths[0]);
  xfree(paths[1]);
  xfree(paths);
  return true;
}

static bool
train_data_cb(LList* node, uint32_t index, void* userdata)
{
  FILE* fp;
  unsigned char header[8]; // 8 is the maximum size that can be checked

  png_structp png_ptr;
  png_infop info_ptr;

  int width, height, row_stride;
  size_t n_pixels;

  png_bytep *input_rows;
  png_bytep input_data;

  char** paths = (char**)node->data;
  char* label_path = paths[0];
  char* depth_path = paths[1];
  TrainData *data = (TrainData*)userdata;

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

  // Verify width/height
  if (data->width == 0)
    {
      data->width = width;
      data->height = height;
      n_pixels = width * height * data->n_images;
      data->label_images = (uint8_t*)xmalloc(n_pixels * sizeof(uint8_t));
      data->depth_images = (float*)xmalloc(n_pixels * sizeof(float));
    }
  else
    {
      if (width != data->width || height != data->height)
        {
          fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
                  label_path, width, height, data->width, data->height);
          exit(1);
        }
    }

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
  row_stride = png_get_rowbytes(png_ptr, info_ptr);
  input_rows = (png_bytep *)xmalloc(sizeof(png_bytep*) * height);
  input_data = (png_bytep)xmalloc(row_stride * height * sizeof(png_bytep));

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

  // Close the label file
  if (fclose(fp) != 0)
    {
      fprintf(stderr, "Error closing label file '%s'\n", label_path);
      exit(1);
    }

  // Free data associated with PNG reading
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  xfree(input_rows);
  xfree(input_data);

  // Read depth file
  float* dest;
  InputFile in_file(depth_path);
  Box2i dw = in_file.header().dataWindow();

  width = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;

  if (width != data->width || height != data->height)
    {
      fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
              depth_path, width, height, data->width, data->height);
      exit(1);
    }

  const ChannelList& channels = in_file.header().channels();
  if (!channels.findChannel("Y"))
    {
      fprintf(stderr, "%s: Only expected to load greyscale EXR files "
              "with a single 'Y' channel\n", depth_path);
      exit(1);
    }

  dest = &data->depth_images[index * width * height];

  FrameBuffer frameBuffer;
  frameBuffer.insert("Y",
                     Slice (FLOAT,
                            (char *)dest,
                            sizeof(float), // x stride,
                            sizeof(float) * width)); // y stride

  in_file.setFrameBuffer(frameBuffer);
  in_file.readPixels(dw.min.y, dw.max.y);

  // Free file path copies
  free_train_data_cb(node, index, userdata);

  return true;
}

void
gather_train_data(const char* label_dir_path, const char* depth_dir_path,
                  uint32_t limit, bool shuffle,
                  uint32_t* out_n_images,
                  int32_t* out_width, int32_t* out_height,
                  float** out_depth_images, uint8_t** out_label_images)
{
  TrainData data = { 0, limit, shuffle, NULL, 0, 0, NULL, NULL };

  gather_train_files(label_dir_path, depth_dir_path, &data);

  *out_n_images = (data.n_images < data.limit) ? data.n_images : data.limit;
  printf("Loading %d training image pairs into memory...\n", *out_n_images);

  data.image_paths = llist_first(data.image_paths);
  if (data.shuffle)
    {
      data.image_paths = llist_slice(llist_shuffle(data.image_paths),
                                     0, data.limit, free_train_data_cb, NULL);
    }
  llist_foreach(data.image_paths, train_data_cb, (void*)&data);
  llist_free(data.image_paths, NULL, NULL);

  *out_width = data.width;
  *out_height = data.height;
  *out_depth_images = data.depth_images;
  *out_label_images = data.label_images;
}

