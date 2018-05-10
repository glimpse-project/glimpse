/*
 * Copyright (C) 2018 Glimp IP Ltd
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <getopt.h>

#include <vector>

#include "half.hpp"

#include "train_utils.h"
#include "loader.h"
#include "infer.h"

#include <glimpse_rdt.h>

using half_float::half;

static bool verbose_opt = false;

static void
logger_cb(struct gm_logger *logger,
          enum gm_log_level level,
          const char *context,
          struct gm_backtrace *backtrace,
          const char *format,
          va_list ap,
          void *user_data)
{
    FILE *log_fp = stderr;
    char *msg = NULL;

    if (verbose_opt == false && level < GM_LOG_ERROR)
        return;

    if (vasprintf(&msg, format, ap) > 0) {
        switch (level) {
        case GM_LOG_ERROR:
            fprintf(log_fp, "%s: ERROR: ", context);
            break;
        case GM_LOG_WARN:
            fprintf(log_fp, "%s: WARN: ", context);
            break;
        default:
            fprintf(log_fp, "%s: ", context);
        }

        fprintf(log_fp, "%s\n", msg);

        if (backtrace) {
            int line_len = 100;
            char *formatted = (char *)alloca(backtrace->n_frames * line_len);

            gm_logger_get_backtrace_strings(logger, backtrace,
                                            line_len, (char *)formatted);
            for (int i = 0; i < backtrace->n_frames; i++) {
                char *line = formatted + line_len * i;
                fprintf(log_fp, "> %s\n", line);
            }
        }

        fflush(log_fp);
        free(msg);
    }
}

static void
logger_abort_cb(struct gm_logger *logger, void *user_data)
{
    FILE *log_fp = stderr;

    fprintf(log_fp, "ABORT\n");
    fflush(log_fp);
    fclose(log_fp);

    abort();
}

static void
usage(void)
{
    fprintf(stderr,
"Usage: test_rdt [OPTIONS] <data dir> <index name> <tree0> [tree1...]\n"
"\n"
"Tests the performance of one or more randomized decision trees across a\n"
"given index of images.\n"
"\n"
"  -v, --verbose                 Verbose output.\n"
"  -h, --help                    Display this message.\n"
    );
    exit(1);
}

int
main(int argc, char **argv)
{
    struct gm_logger *log = gm_logger_new(logger_cb, NULL);
    gm_logger_set_abort_callback(log, logger_abort_cb, NULL);

    const char *short_options="vh";
    const struct option long_options[] = {
        {"verbose",         no_argument,        0, 'v'},
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
        case 'v':
            verbose_opt = true;
            break;
        case 'h':
            usage();
            break;
        default:
            usage();
            break;
        }
    }

    if (argc - optind < 3)
        usage();

    const char *data_dir = argv[optind];
    const char *index_name = argv[optind + 1];

    int n_trees = argc - optind - 2;
    char *tree_paths[n_trees];
    for (int i = 0; i < n_trees; i++) {
        tree_paths[i] = argv[optind + 2 + i];
    }

    RDTree **forest = read_forest((const char**)tree_paths, n_trees);
    if (!forest) {
        fprintf(stderr, "Failed to load decision tree[s]\n");
        exit(1);
    }

    int width;
    int height;
    int n_images;
    int n_labels;

    half *depth_images;
    uint8_t *label_images;

    char *err = NULL;

    if (!gather_train_data(log,
                           data_dir,
                           index_name,
                           NULL, // no joint map
                           &n_images,
                           NULL, // n_joints
                           &width,
                           &height,
                           &depth_images,
                           &label_images,
                           NULL, // no joint data
                           &n_labels,
                           NULL, // fov
                           &err))
    {
        return false;
    }

    float *probs = (float*)xmalloc(width * height * sizeof(float) * n_labels);

    int label_incidence[n_labels];
    int correct_label_inference[n_labels];

    float best_accuracy = 0;
    float worst_accuracy = 1.0;
    float average_accuracy = 0;

    std::vector<float> all_accuracies;
    all_accuracies.reserve(n_images);

    for (int i = 0; i < n_images; i++) {
        infer_labels<half>(forest,
                           n_trees,
                           &depth_images[i * width * height],
                           width,
                           height,
                           probs);

        uint8_t *labels = &label_images[i * width * height];

        memset(label_incidence, 0, sizeof(label_incidence));
        memset(correct_label_inference, 0, sizeof(correct_label_inference));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int off = y * width + x;

                int actual_label = (int)labels[off];

                float *pr_table = &probs[off * n_labels];
                uint8_t inferred_label = 0;
                float pr = -1.0;
                for (int l = 0; l < n_labels; l++) {
                    if (pr_table[l] > pr) {
                        inferred_label = l;
                        pr = pr_table[l];
                    }
                }

                label_incidence[actual_label]++;
                if (inferred_label == actual_label) {
                    correct_label_inference[inferred_label]++;
                }
            }
        }

        int present_labels = 0;
        float accuracy = 0.f;
        for (int l = 0; l < n_labels; l++) {
            if (label_incidence[l] > 0) {
                accuracy += correct_label_inference[l] /
                    (float)label_incidence[l];
                present_labels++;
            }
        }
        accuracy /= (float)present_labels;

        if (accuracy < worst_accuracy)
            worst_accuracy = accuracy;
        if (accuracy > best_accuracy)
            best_accuracy = accuracy;
        all_accuracies.push_back(accuracy);

        average_accuracy += accuracy;
    }

    average_accuracy /= n_images;
    std::sort(all_accuracies.begin(), all_accuracies.end());

    printf("Accuracy across '%s' index of %d images:\n",
           index_name, (int)all_accuracies.size());
    printf("Average: %.2f\n", average_accuracy);
    printf("Median:  %.2f\n", all_accuracies[all_accuracies.size() / 2]);
    printf("Worst:   %.2f\n", worst_accuracy);
    printf("Best:    %.2f\n", best_accuracy);

    printf("Histogram:\n");
    int histogram_len = 30;
    int histogram[histogram_len];
    memset(histogram, 0, sizeof(histogram));

    int max_entries = 0;
    for (int i = 0; i < (int)all_accuracies.size(); i++) {
        int bucket = all_accuracies[i] * histogram_len;
        histogram[bucket]++;
        if (histogram[bucket] > max_entries) {
            max_entries = histogram[bucket];
        }
    }

    static const char *bars[] = {
        " ",
        "▏",
        "▎",
        "▍",
        "▌",
        "▋",
        "▊",
        "▉",
        "█"
    };

    int max_bar_width = 30; // measured in terminal columns

    for (int i = 0; i < histogram_len; i++) {
        int bar_len = max_bar_width * 8 * histogram[i] / max_entries;
        printf("%-3d%%|", (int)(((float)i/histogram_len) * 100.0f));
        for (int j = 0; j < max_bar_width; j++) {
            if (bar_len > 8) {
                printf("%s", bars[8]);
                bar_len -= 8;
            } else {
                printf("%s", bars[bar_len]);
                bar_len = 0;
            }
        }
        printf("| %d\n", histogram[i]);
    }

    return 0;
}
