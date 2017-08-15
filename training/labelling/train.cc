#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <dirent.h>
#include <stdint.h>
#include <stdbool.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <random>
#include <thread>
#include <pthread.h>
#include <time.h>

#include <png.h>

#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImathBox.h>

#include "xalloc.h"
#include "llist.h"
#include "utils.h"

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

static bool verbose = false;

typedef bool (*TrainDataCallback)(const char* label_path,
                                  const char* depth_path,
                                  void*       userdata);

typedef struct {
  int32_t hours;
  int32_t minutes;
  int32_t seconds;
} TimeForDisplay;

typedef struct {
  int32_t  width;         // Width of training images
  int32_t  height;        // Height of training images
  float    ppm;           // Pixels per meter
  uint8_t  n_labels;      // Number of labels in label images

  uint32_t n_images;      // Number of training images
  uint8_t* label_images;  // Label images (row-major)
  float*   depth_images;  // Depth images (row-major)

  uint32_t n_uv;          // Number of combinations of u,v pairs
  float    uv_range;      // Range of u,v combinations to generate
  uint32_t n_t;           // The number of thresholds
  float    t_range;       // Range of thresholds to test
  uint8_t  max_depth;     // Maximum depth to train to
  uint32_t n_pixels;      // Number of pixels to sample
  UVPair*  uvs;           // A list of uv pairs to test
  float*   ts;            // A list of thresholds to test
} TrainContext;

typedef struct {
  uint32_t n_images;      // Number of training images
  uint32_t limit;         // Limit to number of training images
  bool     shuffle;       // Whether to shuffle images
  LList*   image_paths;   // List of label and depth file paths
} TrainData;

typedef struct {
  uint32_t  id;              // Unique id to place the node a tree.
  uint32_t  depth;           // Tree depth at which this node sits.
  uint32_t* pixel_base;      // Index at which a particular image's pixels starts
  Int2D*    pixels;          // A list of pixel pairs.
} NodeTrainData;

typedef struct {
  TrainContext*      ctx;                // The context to use
  NodeTrainData**    data;               // The node data to use and modify
  uint32_t           c_start;            // The uv combination to start on
  uint32_t           c_end;              // The uv combination to end on
  float*             root_nhistogram;    // Normalised histogram of labels
  float*             best_gain;          // Best gain achieved
  uint32_t*          best_uv;            // Index of the best uv combination
  uint32_t*          best_t;             // Index of the best threshold
  pthread_barrier_t* ready_barrier;      // Barrier to wait on to start work
  pthread_barrier_t* finished_barrier;   // Barrier to wait on when finished
} TrainThreadData;

static bool
gather_train_data(const char* label_dir_path,
                  const char* depth_dir_path,
                  TrainDataCallback train_cb,
                  void* userdata)
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
          cont = gather_train_data(next_label_path, next_depth_path, train_cb,
                                   userdata);
        }
      else if ((ext = strstr(label_entry->d_name, ".png")) && ext[4] == '\0')
        {
          strcpy(next_depth_path + strlen(next_depth_path) - 4, ".exr");
          cont = train_cb(next_label_path, next_depth_path, userdata);
        }

      free(next_label_path);
      free(next_depth_path);
    }

  closedir(label_dir);

  return cont;
}

static bool
gather_cb(const char* label_path, const char* depth_path, void* userdata)
{
  TrainData* data = (TrainData*)userdata;
  char** paths = (char**)xmalloc(2 * sizeof(char*));
  paths[0] = strdup(label_path);
  paths[1] = strdup(depth_path);
  data->image_paths = llist_insert_after(data->image_paths, llist_new(paths));
  ++data->n_images;
  return data->shuffle || (data->n_images < data->limit);
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
  TrainContext *ctx = (TrainContext*)userdata;

  if (verbose)
    {
      printf("%06u - %s\n         %s\n", index, depth_path, label_path);
    }

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
  if (ctx->width == 0)
    {
      ctx->width = width;
      ctx->height = height;
      n_pixels = width * height * ctx->n_images;
      ctx->label_images = (uint8_t*)xmalloc(n_pixels * sizeof(uint8_t));
      ctx->depth_images = (float*)xmalloc(n_pixels * sizeof(float));
    }
  else
    {
      if (width != ctx->width || height != ctx->height)
        {
          fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
                  label_path, width, height, ctx->width, ctx->height);
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
      memcpy(&ctx->label_images[dest_idx], &input_data[src_idx], width);
    }

  // Close the label file
  if (fclose(fp) != 0)
    {
      fprintf(stderr, "Error closing label file '%s'\n", label_path);
      exit(1);
    }

  // Read depth file
  float* dest;
  InputFile in_file(depth_path);
  Box2i dw = in_file.header().dataWindow();

  width = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;

  if (width != ctx->width || height != ctx->height)
    {
      fprintf(stderr, "%s: size mismatch (%dx%d), expected (%dx%d)\n",
              depth_path, width, height, ctx->width, ctx->height);
      exit(1);
    }

  const ChannelList& channels = in_file.header().channels();
  if (!channels.findChannel("Y"))
    {
      fprintf(stderr, "%s: Only expected to load greyscale EXR files "
              "with a single 'Y' channel\n", depth_path);
      exit(1);
    }

  dest = &ctx->depth_images[index * width * height];

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

static NodeTrainData*
create_node_train_data(TrainContext* ctx, uint32_t id, uint32_t depth,
                       uint32_t* pixel_base, Int2D* pixels)
{
  NodeTrainData* data = (NodeTrainData*)xcalloc(1, sizeof(NodeTrainData));

  data->id = id;
  data->depth = depth;

  if (pixel_base)
    {
      data->pixel_base = pixel_base;
    }
  else
    {
      // Assume this is the root node and each image has n_pixels
      size_t size_pixel_base = (ctx->n_images + 1) * sizeof(uint32_t);
      data->pixel_base = (uint32_t*)xmalloc(size_pixel_base);
      for (uint32_t i = 0; i <= ctx->n_images; i++)
        {
          data->pixel_base[i] = i * ctx->n_pixels;
        }
    }

  if (pixels)
    {
      data->pixels = pixels;
    }
  else
    {
      // Assume this is the root node and generate random coordinates
      uint32_t total_pixels = ctx->n_images * ctx->n_pixels;
      data->pixels = (Int2D*)xmalloc(total_pixels * sizeof(Int2D));

      std::random_device rd;
      std::mt19937 rng(rd());
      std::uniform_int_distribution<int> rand_x(0, ctx->width - 1);
      std::uniform_int_distribution<int> rand_y(0, ctx->height - 1);
      for (uint32_t i = 0; i < total_pixels; i++)
        {
          data->pixels[i][0] = rand_x(rng);
          data->pixels[i][1] = rand_y(rng);
        }
    }

  return data;
}

static void
destroy_node_train_data(NodeTrainData* data)
{
  xfree(data->pixels);
  xfree(data->pixel_base);
  xfree(data);
}

static Int2D
normalize_histogram(uint32_t* histogram, uint8_t n_labels, float* normalized)
{
  Int2D sums = { 0, 0 };

  for (int i = 0; i < n_labels; i++)
    {
      if (histogram[i] > 0)
        {
          sums[0] += histogram[i];
          ++sums[1];
        }
    }

  if (sums[0] > 0)
    {
      for (int i = 0; i < n_labels; i++)
        {
          normalized[i] = histogram[i] / (float)sums[0];
        }
    }
  else
    {
      memset(normalized, 0, n_labels * sizeof(float));
    }

  return sums;
}

static float
calculate_shannon_entropy(float* normalized_histogram, uint8_t n_labels)
{
  float entropy = 0.f;
  for (int i = 0; i < n_labels; i++)
    {
      float value = normalized_histogram[i];
      if (value > 0.f)
        {
          float add = -value * log2f(value);
          if (std::isnormal(add))
            {
              entropy += add;
            }
        }
    }
  return entropy;
}

static float
calculate_gain(float entropy, uint32_t n_pixels,
               float l_entropy, uint32_t l_n_pixels,
               float r_entropy, uint32_t r_n_pixels)
{
  return entropy - ((l_n_pixels / (float)n_pixels * l_entropy) +
                    (r_n_pixels / (float)n_pixels * r_entropy));
}

static void
accumulate_histograms(TrainContext* ctx, NodeTrainData* data,
                      uint32_t c_start, uint32_t c_end,
                      uint32_t* root_histogram, uint32_t* lr_histograms)
{
  uint32_t image_idx = 0;
  for (uint32_t i = 0; i < ctx->n_images;
       i++, image_idx += ctx->width * ctx->height)
    {
      float* depth_image = &ctx->depth_images[image_idx];
      uint8_t* label_image = &ctx->label_images[image_idx];

      for (uint32_t p = data->pixel_base[i]; p < data->pixel_base[i + 1]; p++)
        {
          Int2D pixel = data->pixels[p];
          uint32_t pixel_idx = (pixel[1] * ctx->width) + pixel[0];
          uint8_t label = label_image[pixel_idx];
          float depth = depth_image[pixel_idx];

          if (label >= ctx->n_labels)
            {
              fprintf(stderr, "Label '%u' is bigger than expected (max %u)\n",
                      (uint32_t)label, (uint32_t)ctx->n_labels - 1);
              exit(1);
            }

          // Accumulate root histogram
          ++root_histogram[label];

          for (uint32_t c = c_start, lr_histogram_idx = 0; c < c_end; c++)
            {
              UVPair uv = ctx->uvs[c];
              float value = sample_uv(depth_image,
                                      ctx->width, ctx->height,
                                      pixel, depth, uv);
              for (uint32_t t = 0; t < ctx->n_t;
                   t++, lr_histogram_idx += ctx->n_labels * 2)
                {
                  // Accumulate histogram for this particular uvt combination
                  // on both theoretical branches
                  float threshold = ctx->ts[t];
                  ++lr_histograms[value < threshold ?
                    lr_histogram_idx + label :
                    lr_histogram_idx + ctx->n_labels + label];
                }
            }
        }
    }
}

static void*
thread_body(void* userdata)
{
  TrainThreadData* data = (TrainThreadData*)userdata;

  // Histogram for the node being processed
  uint32_t* root_histogram = (uint32_t*)
    malloc(data->ctx->n_labels * sizeof(uint32_t));

  // Histograms for each uvt combination being tested
  uint32_t* lr_histograms = (uint32_t*)
    malloc(data->ctx->n_labels * (data->c_end - data->c_start) *
           data->ctx->n_t * 2 * sizeof(uint32_t));

  float* nhistogram = (float*)xmalloc(data->ctx->n_labels * sizeof(float));
  float* root_nhistogram = data->root_nhistogram ? data->root_nhistogram :
    (float*)xmalloc(data->ctx->n_labels * sizeof(float));

  while (1)
    {
      // Wait for everything to be ready to start processing
      pthread_barrier_wait(data->ready_barrier);

      // Quit out if we've nothing left to process
      if (!(*data->data))
        {
          break;
        }

      // Clear histogram accumulators
      memset(root_histogram, 0, data->ctx->n_labels * sizeof(uint32_t));
      memset(lr_histograms, 0, data->ctx->n_labels *
             (data->c_end - data->c_start) * data->ctx->n_t * 2 *
             sizeof(uint32_t));

      // Accumulate histograms
      accumulate_histograms(data->ctx, *data->data, data->c_start, data->c_end,
                            root_histogram, lr_histograms);

      // Calculate the normalised label histogram and get the number of pixels
      // and the number of labels in the root histogram.
      Int2D root_n_pixels = normalize_histogram(root_histogram,
                                                data->ctx->n_labels,
                                                root_nhistogram);

      // Determine the best u,v,t combination
      *data->best_gain = 0.f;

      // If there's only 1 label, skip all this, gain is zero
      if (root_n_pixels[1] > 1)
        {
          // Calculate the shannon entropy for the normalised label histogram
          float entropy = calculate_shannon_entropy(root_nhistogram,
                                                    data->ctx->n_labels);

          // Calculate the gain for each combination of u,v,t and store the best
          for (uint32_t i = data->c_start, lr_histo_base = 0;
               i < data->c_end; i++)
            {
              for (uint32_t j = 0; j < data->ctx->n_t;
                   j++, lr_histo_base += data->ctx->n_labels * 2)
                {
                  float l_entropy, r_entropy, gain;

                  Int2D l_n_pixels =
                    normalize_histogram(&lr_histograms[lr_histo_base],
                                        data->ctx->n_labels, nhistogram);
                  if (l_n_pixels[0] == 0 || l_n_pixels[0] == root_n_pixels[0])
                    {
                      continue;
                    }
                  l_entropy = calculate_shannon_entropy(nhistogram,
                                                        data->ctx->n_labels);

                  Int2D r_n_pixels =
                    normalize_histogram(
                      &lr_histograms[lr_histo_base + data->ctx->n_labels],
                      data->ctx->n_labels, nhistogram);
                  r_entropy = calculate_shannon_entropy(nhistogram,
                                                        data->ctx->n_labels);

                  gain = calculate_gain(entropy, root_n_pixels[0],
                                        l_entropy, l_n_pixels[0],
                                        r_entropy, r_n_pixels[0]);

                  if (gain > *data->best_gain)
                    {
                      *data->best_gain = gain;
                      *data->best_uv = i;
                      *data->best_t = j;
                    }
                }
            }
        }

      // Signal work is finished
      pthread_barrier_wait(data->finished_barrier);
    }

  xfree(root_histogram);
  xfree(lr_histograms);
  if (!data->root_nhistogram)
    {
      xfree(root_nhistogram);
    }
  xfree(nhistogram);
  xfree(data);

  pthread_exit(NULL);
}

static void
collect_pixels(TrainContext* ctx, NodeTrainData* data, UVPair uv, float t,
               uint32_t** l_pixel_base, Int2D** l_pixels,
               uint32_t** r_pixel_base, Int2D** r_pixels)
{
  *l_pixel_base = (uint32_t*)xcalloc(ctx->n_images + 1, sizeof(uint32_t));
  *r_pixel_base = (uint32_t*)xcalloc(ctx->n_images + 1, sizeof(uint32_t));

  // Start off allocating these as large as they could be, we'll reallocate
  // after we know the sizes from the loop
  *l_pixels = (Int2D*)xmalloc(data->pixel_base[ctx->n_images] * sizeof(Int2D));
  *r_pixels = (Int2D*)xmalloc(data->pixel_base[ctx->n_images] * sizeof(Int2D));

  uint32_t image_idx = 0;
  for (uint32_t i = 0; i < ctx->n_images; i++)
    {
      float* depth_image = &ctx->depth_images[image_idx];
      for (uint32_t p = data->pixel_base[i]; p < data->pixel_base[i + 1]; p++)
        {
          Int2D pixel = data->pixels[p];
          float depth = depth_image[(pixel[1] * ctx->width) + pixel[0]];
          float value = sample_uv(depth_image, ctx->width, ctx->height,
                                  pixel, depth, uv);

          if (value < t)
            {
              (*l_pixels)[(*l_pixel_base)[i + 1]] = pixel;
              ++(*l_pixel_base)[i + 1];
            }
          else
            {
              (*r_pixels)[(*r_pixel_base)[i + 1]] = pixel;
              ++(*r_pixel_base)[i + 1];
            }
        }

      if (i < ctx->n_images - 1)
        {
          (*l_pixel_base)[i + 2] = (*l_pixel_base)[i + 1];
          (*r_pixel_base)[i + 2] = (*r_pixel_base)[i + 1];
        }
      image_idx += ctx->width * ctx->height;
    }

  // Shrink pixel arrays down to the correct size
  *l_pixels = (Int2D*)xrealloc((void*)(*l_pixels),
                               (*l_pixel_base)[ctx->n_images] * sizeof(Int2D));
  *r_pixels = (Int2D*)xrealloc((void*)(*r_pixels),
                               (*r_pixel_base)[ctx->n_images] * sizeof(Int2D));
}

static bool
list_free_cb(LList* node, uint32_t index, void* userdata)
{
  xfree(node->data);
  return true;
}

static void
print_usage(FILE* stream)
{
  fprintf(stream,
"Usage: train <ppm> <n_labels> <label_dir> <depth_dir> <out_file> [OPTIONS]\n"
"Train a randomised decision tree to infer n_labels from depth images with\n"
"ppm pixels per meter.\n"
"\n"
"  -l, --limit=NUMBER        Limit training data to this many images\n"
"  -s, --shuffle             Shuffle order of training images\n"
"  -p, --pixels=NUMBER       Number of pixels to sample per image (default: 2000)\n"
"  -t, --thresholds=NUMBER   Number of thresholds to test (default: 50)\n"
"  -r, --t-range=NUMBER      Range of thresholds to test (default: 1.29)\n"
"  -c, --combos=NUMBER       Number of UV combinations to test (default: 2000)\n"
"  -u, --uv-range=NUMBER     Range of UV combinations to test (default 1.29)\n"
"  -d, --depth=NUMBER        Depth to train tree to (default: 20)\n"
"  -m, --threads=NUMBER      Number of threads to use (default: autodetect)\n"
"  -v, --verbose             Verbose output\n"
"  -h, --help                Display this message\n");
}

TimeForDisplay
get_time_for_display(struct timespec* begin, struct timespec* end)
{
  uint32_t elapsed;
  TimeForDisplay display;

  elapsed = (end->tv_sec - begin->tv_sec);
  elapsed += (end->tv_nsec - begin->tv_nsec) / 1000000000;

  display.seconds = elapsed % 60;
  display.minutes = elapsed / 60;
  display.hours = display.minutes / 60;
  display.minutes = display.minutes % 60;

  return display;
}

int
main(int argc, char **argv)
{
  TrainContext ctx = { 0, };
  TrainData data = { 0, };
  TimeForDisplay since_begin, since_last;
  struct timespec begin, last, now;
  uint32_t n_threads = std::thread::hardware_concurrency();

  if (argc < 6)
    {
      print_usage(stderr);
      exit(1);
    }

  // Set default parameters
  ctx.n_uv = 2000;
  ctx.uv_range = 1.29;
  ctx.n_t = 50;
  ctx.t_range = 1.29;
  ctx.max_depth = 20;
  ctx.n_pixels = 2000;
  data.limit = UINT32_MAX;
  data.shuffle = false;

  for (int i = 6; i < argc; i++)
    {
      char* number_text = NULL;
      uint32_t* uint32_target = NULL;
      uint8_t* uint8_target = NULL;
      float* float_target = NULL;

      if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--shuffle") == 0)
        {
          data.shuffle = true;
          continue;
        }
      if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
          verbose = true;
          continue;
        }

      if (argv[i][0] != '-')
        {
          fprintf(stderr, "Unrecognised option: %s", argv[i]);
          return 1;
        }

      if (argv[i][1] == '-')
        {
          if (strcmp(argv[i], "--limit=") == 0)
            {
              number_text = &argv[i][8];
              uint32_target = &data.limit;
            }
          else if (strcmp(argv[i], "--pixels=") == 0)
            {
              number_text = &argv[i][9];
              uint32_target = &ctx.n_pixels;
            }
          else if (strcmp(argv[i], "--thresholds=") == 0)
            {
              number_text = &argv[i][13];
              uint32_target = &ctx.n_t;
            }
          else if (strcmp(argv[i], "--t-range=") == 0)
            {
              number_text = &argv[i][10];
              float_target = &ctx.t_range;
            }
          else if (strcmp(argv[i], "--combos=") == 0)
            {
              number_text = &argv[i][9];
              uint32_target = &ctx.n_uv;
            }
          else if (strcmp(argv[i], "--uv-range=") == 0)
            {
              number_text = &argv[i][11];
              float_target = &ctx.uv_range;
            }
          else if (strcmp(argv[i], "--depth=") == 0)
            {
              number_text = &argv[i][8];
              uint8_target = &ctx.max_depth;
            }
          else if (strcmp(argv[i], "--threads=") == 0)
            {
              number_text = &argv[i][10];
              uint32_target = &n_threads;
            }
          else if (strcmp(argv[i], "--help") == 0)
            {
              print_usage(stdout);
              return 0;
            }
        }
      else
        {
          switch (argv[i][1])
            {
            case 'l':
              uint32_target = &data.limit;
              break;

            case 'p':
              uint32_target = &ctx.n_pixels;
              break;

            case 't':
              uint32_target = &ctx.n_t;
              break;

            case 'r':
              float_target = &ctx.t_range;
              break;

            case 'c':
              uint32_target = &ctx.n_uv;
              break;

            case 'u':
              float_target = &ctx.uv_range;
              break;

            case 'd':
              uint8_target = &ctx.max_depth;
              break;

            case 'm':
              uint32_target = &n_threads;
              break;

            case 'h':
              print_usage(stdout);
              return 0;

            default:
              fprintf(stderr, "Unrecognised option: %s", argv[i]);
              print_usage(stderr);
              return 1;
            }

          ++i;
          if (i >= argc)
            {
              fprintf(stderr, "Incorrect usage");
              print_usage(stderr);
              return 1;
            }
          number_text = argv[i];
        }

      if (uint32_target)
        {
          *uint32_target = (uint32_t)atoi(number_text);
        }
      else if (uint8_target)
        {
          *uint8_target = (uint8_t)atoi(number_text);
        }
      else if (float_target)
        {
          *float_target = strtof(number_text, NULL);
        }
    }

  ctx.ppm = strtof(argv[1], NULL);
  ctx.uv_range *= ctx.ppm;
  ctx.n_labels = (uint8_t)atoi(argv[2]);

  printf("Opening output file...\n");
  FILE* output;
  if (!(output = fopen(argv[5], "wb")))
    {
      fprintf(stderr, "Failed to open output file '%s'\n", argv[5]);
      exit(1);
    }

  printf("Scanning training directories...\n");
  gather_train_data(argv[3], argv[4], gather_cb, (void*)&data);

  ctx.n_images = (data.n_images < data.limit) ? data.n_images : data.limit;
  printf("Loading %d training image pairs into memory...\n", ctx.n_images);
  data.image_paths = llist_first(data.image_paths);
  if (data.shuffle)
    {
      data.image_paths = llist_slice(llist_shuffle(data.image_paths),
                                     0, data.limit, free_train_data_cb, NULL);
    }
  llist_foreach(data.image_paths, train_data_cb, (void*)&ctx);
  llist_free(data.image_paths, NULL, NULL);
  data.image_paths = NULL;

  // Initialise root node training data and add it to the queue
  printf("Preparing training metadata...\n");
  ctx.uvs = (UVPair*)xmalloc(ctx.n_uv * sizeof(UVPair));
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> rand_uv(-ctx.uv_range / 2.f,
                                                 ctx.uv_range / 2.f);
  for (uint32_t i = 0; i < ctx.n_uv; i++)
    {
      ctx.uvs[i][0] = rand_uv(rng);
      ctx.uvs[i][1] = rand_uv(rng);
      ctx.uvs[i][2] = rand_uv(rng);
      ctx.uvs[i][3] = rand_uv(rng);
    }
  ctx.ts = (float*)xmalloc(ctx.n_t * sizeof(float));
  for (uint32_t i = 0; i < ctx.n_t; i++)
    {
      ctx.ts[i] = -ctx.t_range / 2.f + (i * ctx.t_range / (float)(ctx.n_t - 1));
    }

  // Allocate memory to store the decision tree.
  uint32_t n_nodes = (uint32_t)roundf(powf(2.f, ctx.max_depth)) - 1;
  Node* tree = (Node*)xcalloc(n_nodes, sizeof(Node));
  LList* tree_histograms = NULL;
  uint32_t n_histograms = 0;

  LList* train_queue =
    llist_new(create_node_train_data(&ctx, 0, 0, NULL, NULL));
  float* root_nhistogram = (float*)xmalloc(ctx.n_labels * sizeof(float));

  NodeTrainData* node_data;
  printf("Initialising %u threads...\n", n_threads);
  pthread_barrier_t ready_barrier, finished_barrier;
  if (pthread_barrier_init(&ready_barrier, NULL, n_threads + 1) != 0 ||
      pthread_barrier_init(&finished_barrier, NULL, n_threads + 1) != 0)
    {
      fprintf(stderr, "Error initialising thread barriers\n");
      return 1;
    }
  uint32_t n_c = (ctx.n_uv + n_threads - 1) / n_threads;
  float* best_gains = (float*)malloc(n_threads * sizeof(float));
  uint32_t* best_uvs = (uint32_t*)malloc(n_threads * sizeof(uint32_t));
  uint32_t* best_ts = (uint32_t*)malloc(n_threads * sizeof(uint32_t));
  for (uint32_t i = 0; i < n_threads; i++)
    {
      TrainThreadData* thread_data = (TrainThreadData*)
        xmalloc(sizeof(TrainThreadData));
      thread_data->ctx = &ctx;
      thread_data->data = &node_data;
      thread_data->c_start = i * n_c;
      thread_data->c_end = std::min((i + 1) * n_c, ctx.n_uv);
      thread_data->root_nhistogram = (i == 0) ? root_nhistogram : NULL;
      thread_data->best_gain = &best_gains[i];
      thread_data->best_uv = &best_uvs[i];
      thread_data->best_t = &best_ts[i];
      thread_data->ready_barrier = &ready_barrier;
      thread_data->finished_barrier = &finished_barrier;

      pthread_t thread;
      if (pthread_create(&thread, NULL, thread_body, (void*)thread_data) != 0)
        {
          fprintf(stderr, "Error creating thread\n");
          return 1;
        }
    }

  printf("Beginning training...\n");
  clock_gettime(CLOCK_MONOTONIC, &begin);
  last = begin;
  uint32_t last_depth = UINT32_MAX;
  while (train_queue != NULL)
    {
      uint32_t best_uv;
      uint32_t best_t;
      float best_gain = 0.0;

      LList* current = train_queue;
      node_data = (NodeTrainData*)current->data;

      if (node_data->depth != last_depth)
        {
          clock_gettime(CLOCK_MONOTONIC, &now);
          since_begin = get_time_for_display(&begin, &now);
          since_last = get_time_for_display(&last, &now);
          last = now;
          last_depth = node_data->depth;
          printf("(%02d:%02d:%02d / %02d:%02d:%02d) Training depth %u (%u nodes)\n",
                 since_begin.hours, since_begin.minutes, since_begin.seconds,
                 since_last.hours, since_last.minutes, since_last.seconds,
                 last_depth + 1, llist_length(train_queue));
        }

      // Signal threads to start work
      pthread_barrier_wait(&ready_barrier);

      // Wait for threads to finish
      pthread_barrier_wait(&finished_barrier);

      // See which thread got the best uvt combination
      for (uint32_t i = 0; i < n_threads; i++)
        {
          if (best_gains[i] > best_gain)
            {
              best_gain = best_gains[i];
              best_uv = best_uvs[i];
              best_t = best_ts[i];
            }
        }

      // Add this node to the tree and possible add left/ride nodes to the
      // training queue.
      Node* node = &tree[node_data->id];
      if (best_gain > 0.f && (node_data->depth + 1) < ctx.max_depth)
        {
          node->uv = ctx.uvs[best_uv];
          node->t = ctx.ts[best_t];
          if (verbose)
            {
              printf("  Node (%u)\n"
                     "    Gain: %f\n"
                     "    U: (%f, %f)\n"
                     "    V: (%f, %f)\n"
                     "    T: %f\n",
                     node_data->id, best_gain,
                     node->uv[0], node->uv[1],
                     node->uv[2], node->uv[3],
                     node->t);
            }

          uint32_t* l_pixel_base;
          uint32_t* r_pixel_base;
          Int2D* l_pixels;
          Int2D* r_pixels;

          collect_pixels(&ctx, node_data, node->uv, node->t,
                         &l_pixel_base, &l_pixels,
                         &r_pixel_base, &r_pixels);

          uint32_t id = (2 * node_data->id) + 1;
          uint32_t depth = node_data->depth + 1;
          NodeTrainData* ldata =
            create_node_train_data(&ctx, id, depth, l_pixel_base, l_pixels);
          NodeTrainData* rdata =
            create_node_train_data(&ctx, id + 1, depth, r_pixel_base, r_pixels);

          // Insert nodes into the training queue
          llist_insert_after(
            llist_insert_after(llist_last(train_queue), llist_new(ldata)),
            llist_new(rdata));
        }
      else
        {
          if (verbose)
            {
              printf("  Leaf node (%u)\n", (uint32_t)node_data->id);
              for (int i = 0; i < ctx.n_labels; i++)
                {
                  if (root_nhistogram[i] > 0.f)
                    {
                      printf("    %02d - %f\n", i, root_nhistogram[i]);
                    }
                }
            }

          node->label_pr_idx = ++n_histograms;
          float* node_histogram = (float*)malloc(ctx.n_labels * sizeof(float));
          memcpy(node_histogram, root_nhistogram, ctx.n_labels * sizeof(float));
          tree_histograms = llist_insert_after(tree_histograms,
                                               llist_new(node_histogram));
        }

      // Remove this node from the queue
      train_queue = train_queue->next;
      llist_free(llist_remove(current), NULL, NULL);

      // We no longer need the train data, free it
      destroy_node_train_data(node_data);
    }

  // Signal threads to free memory and quit
  node_data = NULL;
  pthread_barrier_wait(&ready_barrier);

  // Free memory that isn't needed anymore
  xfree(root_nhistogram);
  xfree(ctx.uvs);
  xfree(ctx.ts);
  xfree(best_gains);
  xfree(best_uvs);
  xfree(best_ts);

  // Restore tree histograms list pointer
  tree_histograms = llist_first(tree_histograms);

  // Write to file
  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Writing output to '%s'...\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds,
         argv[5]);

  // Write a header (3 bytes 'RDT', 1 byte version, 1 byte depth, 1 byte n_labels)
  RDLHeader header = { { 'R', 'D', 'T' }, OUT_VERSION, ctx.max_depth, ctx.n_labels };
  if (fwrite(&header, sizeof(RDLHeader), 1, output) != 1)
    {
      fprintf(stderr, "Error writing header\n");
      return 1;
    }
  if (fwrite(tree, sizeof(Node), n_nodes, output) != n_nodes)
    {
      fprintf(stderr, "Error writing tree\n");
      return 1;
    }
  for (LList* l = tree_histograms; l; l = l->next)
    {
      if (fwrite(l->data, sizeof(float), ctx.n_labels, output) != ctx.n_labels)
        {
          fprintf(stderr, "Error writing labels\n");
          return 1;
        }
    }

  // Close the output file
  if (fclose(output) != 0)
    {
      fprintf(stderr, "Error closing output file\n");
      return 1;
    }

  // Free the last data
  xfree(tree);
  llist_free(tree_histograms, list_free_cb, NULL);

  clock_gettime(CLOCK_MONOTONIC, &now);
  since_begin = get_time_for_display(&begin, &now);
  since_last = get_time_for_display(&last, &now);
  last = now;
  printf("(%02d:%02d:%02d / %02d:%02d:%02d) Done!\n",
         since_begin.hours, since_begin.minutes, since_begin.seconds,
         since_last.hours, since_last.minutes, since_last.seconds);

  return 0;
}
