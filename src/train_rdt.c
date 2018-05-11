/*
 * Copyright (C) 2017 Glimp IP Ltd
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

#define _GNU_SOURCE
#include <stdio.h>
#include <getopt.h>

#include <glimpse_rdt.h>
#include <glimpse_config.h>

struct training_data {
    FILE *log_fp;
    struct gm_logger *log;
    struct gm_rdt_context *ctx;
};

static void
logger_cb(struct gm_logger *logger,
          enum gm_log_level level,
          const char *context,
          struct gm_backtrace *backtrace,
          const char *format,
          va_list ap,
          void *user_data)
{
    struct training_data *data = user_data;
    char *msg = NULL;

    if (vasprintf(&msg, format, ap) > 0) {
        if (data->log_fp) {
            switch (level) {
            case GM_LOG_ERROR:
                fprintf(data->log_fp, "%s: ERROR: ", context);
                break;
            case GM_LOG_WARN:
                fprintf(data->log_fp, "%s: WARN: ", context);
                break;
            default:
                fprintf(data->log_fp, "%s: ", context);
            }

            fprintf(data->log_fp, "%s\n", msg);

            if (backtrace) {
                int line_len = 100;
                char *formatted = (char *)alloca(backtrace->n_frames * line_len);

                gm_logger_get_backtrace_strings(logger, backtrace,
                                                line_len, (char *)formatted);
                for (int i = 0; i < backtrace->n_frames; i++) {
                    char *line = formatted + line_len * i;
                    fprintf(data->log_fp, "> %s\n", line);
                }
            }

            fflush(data->log_fp);
            fflush(stdout);
        }

        free(msg);
    }
}

static void
logger_abort_cb(struct gm_logger *logger, void *user_data)
{
    struct training_data *data = user_data;

    if (data->log_fp) {
        fprintf(data->log_fp, "ABORT\n");
        fflush(data->log_fp);
        fclose(data->log_fp);
    }

    abort();
}

static void
usage(void)
{
    fprintf(stderr,
"Usage: train_rdt [OPTIONS] <data dir> <index name> <out file.json>\n"
"\n"
"Train a randomised decision tree to infer body part labels from a data set of\n"
"depth and label images.\n"
"\n"
"  -p, --pixels=NUMBER           Number of pixels to sample per image.\n"
"                                  (default: 2000)\n"
"  -t, --thresholds=NUMBER       Number of thresholds to test.\n"
"                                  (default: 50)\n"
"  -r, --t-range=NUMBER          Range of thresholds to test (in meters).\n"
"                                  (default: 1.29)\n"
"  -c, --combos=NUMBER           Number of UV combinations to test.\n"
"                                  (default: 2000)\n"
"  -u, --uv-range=NUMBER         Range of UV combinations to test (in meters).\n"
"                                  (default 1.29)\n"
"  -d, --depth=NUMBER            Depth to train tree to.\n"
"                                  (default: 20)\n"
"  -m, --threads=NUMBER          Number of threads to use.\n"
"                                  (default: autodetect)\n"
"  -s, --seed=NUMBER             Seed to use for RNG.\n"
"                                  (default: 0)\n"
"  -i, --continue                Continue training from an interrupted run.\n"
"  -v, --verbose                 Verbose output.\n"
"  -h, --help                    Display this message.\n");
    exit(1);
}

int
main(int argc, char **argv)
{
    struct training_data _data;
    struct training_data *data = &_data;

    data->log_fp = stderr;
    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

    data->ctx = gm_rdt_context_new(data->log);

    struct gm_ui_properties *ctx_props =
        gm_rdt_context_get_ui_properties(data->ctx);
    int opt;

    const char *short_options="p:t:r:c:u:d:m:b:s:ivh";
    const struct option long_options[] = {
        {"pixels",          required_argument,  0, 'p'},
        {"thresholds",      required_argument,  0, 't'},
        {"t-range",         required_argument,  0, 'r'},
        {"combos",          required_argument,  0, 'c'},
        {"uv-range",        required_argument,  0, 'u'},
        {"depth",           required_argument,  0, 'd'},
        {"threads",         required_argument,  0, 'm'},
        {"seed",            required_argument,  0, 's'},
        {"continue",        no_argument,        0, 'i'},
        {"verbose",         no_argument,        0, 'v'},
        {"help",            no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        switch (opt) {
        case 'p':
            gm_props_set_int(ctx_props, "n_pixels", atoi(optarg));
            break;
        case 't':
            gm_props_set_int(ctx_props, "n_thresholds", atoi(optarg));
            break;
        case 'r':
            gm_props_set_float(ctx_props, "threshold_range", strtof(optarg, NULL));
            break;
        case 'c':
            gm_props_set_int(ctx_props, "n_uv", atoi(optarg));
            break;
        case 'u':
            gm_props_set_float(ctx_props, "uv_range", strtof(optarg, NULL));
            break;
        case 'd':
            gm_props_set_int(ctx_props, "max_depth", atoi(optarg));
            break;
        case 'm':
            gm_props_set_int(ctx_props, "n_threads", atoi(optarg));
            break;
        case 's':
            gm_props_set_int(ctx_props, "seed", atoi(optarg));
            break;
        case 'i':
            gm_props_set_bool(ctx_props, "reload", true);
            break;
        case 'v':
            gm_props_set_bool(ctx_props, "verbose", true);
            break;
        case 'h':
            usage();
            break;
        default:
            usage();
            break;
        }
    }

    if (optind != argc - 3)
        usage();

    gm_props_set_string(ctx_props, "data_dir", argv[optind]);
    gm_props_set_string(ctx_props, "index_name", argv[optind + 1]);
    gm_props_set_string(ctx_props, "out_file", argv[optind + 2]);


    char *ext = strstr(argv[optind + 2], ".rdt");
    if (ext && strlen(ext) == 4) {
        fprintf(stderr,
"ERROR: Saving to a .rdt file is no longer supported because an RDT file only\n"
"       needs to support the minimum necessary for efficient runtime inference\n"
"       which limits our ability to preserve arbitrary ancillary meta data in\n"
"       our results\n"
"\n"
"Note: A packed RDT file can be created afterwards with json-to-rdt\n"
"\n"
                );
        usage();
    }

    char *err = NULL;
    if (!gm_rdt_context_train(data->ctx, &err)) {
        fprintf(stderr, "Failed to run training: %s\n", err);
        free(err);
    }

    gm_rdt_context_destroy(data->ctx);

    return 0;
}
