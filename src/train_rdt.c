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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <getopt.h>

#include <glimpse_rdt.h>
#include <glimpse_properties.h>

#include "xalloc.h"

#undef GM_LOG_CONTEXT
#define GM_LOG_CONTEXT "train_rdt"

struct training_data {
    FILE *log_fp;
    struct gm_logger *log;
};

static bool verbose_opt = false;
static bool continue_opt = false;
static char *data_dir_opt = NULL;
static char *index_name_opt = NULL;
static char *out_file_opt = NULL;
static int n_threads_opt = 0;
static int seed_opt = 0;

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

    if (verbose_opt == false && level < GM_LOG_INFO)
        return;

    if (vasprintf(&msg, format, ap) > 0) {
        switch (level) {
        case GM_LOG_ERROR:
            fprintf(data->log_fp, "%s: ERROR: ", context);
            if (data->log_fp != stderr)
                fprintf(stderr, "%s: ERROR: ", context);
            break;
        case GM_LOG_WARN:
            fprintf(data->log_fp, "%s: WARN: ", context);
            if (data->log_fp != stderr)
                fprintf(stderr, "%s: WARN: ", context);
            break;
        default:
            fprintf(data->log_fp, "%s: ", context);
        }

        fprintf(data->log_fp, "%s\n", msg);
        if (level >= GM_LOG_WARN && data->log_fp != stderr)
            fprintf(stderr, "%s\n", msg);

        if (backtrace) {
            int line_len = 100;
            char *formatted = (char *)alloca(backtrace->n_frames * line_len);

            gm_logger_get_backtrace_strings(logger, backtrace,
                                            line_len, (char *)formatted);
            for (int i = 0; i < backtrace->n_frames; i++) {
                char *line = formatted + line_len * i;
                fprintf(data->log_fp, "> %s\n", line);
                if (data->log_fp != stderr)
                    fprintf(stderr, "> %s\n", line);
            }
        }

        fflush(data->log_fp);
        if (data->log_fp != stderr)
            fflush(stderr);

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
usage(struct gm_ui_properties *ctx_props)
{
    fprintf(stderr,
"Usage:\n"
"  train_rdt [options] -q <run-queue.json> [<index_name>] [<results.json>]\n"
"  train_rdt [options] -q '[{ \"prop\": \"value\", ...}, ...]' \\\n"
"              [<index_name>] [<results.json>]\n"
"  train_rdt [options] <index_name> <results.json>\n"
"\n"
"Train one or more randomised decision trees to infer body part labels from a\n"
"data set of depth and label images.\n"
"\n"
"Options:\n"
"\n"
"  -d, --data-dir=DIR         Path to training data.\n"
"                             (current directory by default)\n"
"\n"
"    NOTE: The path is prepended to any relative \"data_dir\" values.\n"
"\n"
"  -q,--queue=QUEUE_JSON      A JSON string or filename containing a queue\n"
"                             of training work in this schema:\n"
"    [\n"
"      {\n"
"        // Run 0...\n"
"        \"index_name\": \"tree0\",\n"
"        \"n_pixels\": 2000,\n"
"        \"n_thresholds\": 50,\n"
"        // ...\n"
"        \"out_file\": \"tree0.json\"\n"
"      },\n"
"      {\n"
"        // Run 1...\n"
"        \"index_name\": \"tree1\",\n"
"        \"n_pixels\": 1000,\n"
"        \"n_thresholds\": 100,\n"
"        // ...\n"
"        \"out_file\": \"tree1.json\"\n"
"      },\n"
"      {\n"
"        // Run 2...\n"
"        \"index_name\": \"tree2\",\n"
"        \"n_pixels\": 500,\n"
"        \"n_thresholds\": 200,\n"
"        // ...\n"
"        \"out_file\": \"tree2.json\"\n"
"      }\n"
"    ]\n"
"\n"
"    NOTE: // or /* */ style comments are allowed.\n"
"    NOTE: Hyperparameters don't automatically carry over between runs.\n"
"    NOTE: --threads, --seed, --verbose, <index_name> and <results.json>\n"
"          effectively set default values for the \"n_threads\", \"seed\",\n"
"          \"verbose\", \"index_name\" and \"out_file\" parameters respectively,\n"
"          and can all be overridden by per-run JSON values.\n"
"    NOTE: Use the -p and -v options to override parameters for all runs.\n"
"\n"
"  -p, --parameter=NAME       Name of hyperparameter to override.\n"
"      --property=NAME\n"
"  -v, --value=VALUE          Value to give hyperparameter.\n"
"\n"
"    NOTE: -p and -v options must be used in pairs\n"
"\n"
);

    fprintf(stderr, "    Available parameters:\n");
    for (int i = 0; i < ctx_props->n_properties; i++) {
        struct gm_ui_property *prop = &ctx_props->properties[i];
        const char *str;

        fprintf(stderr, "    %-15s - %s ", prop->name, prop->desc);
        switch (prop->type) {
        case GM_PROPERTY_INT:
            fprintf(stderr, "\n    %18s(default=%d)\n", "", gm_prop_get_int(prop));
            break;
        case GM_PROPERTY_FLOAT:
            fprintf(stderr, "\n    %18s(default=%.2f)\n", "", gm_prop_get_float(prop));
            break;
        case GM_PROPERTY_BOOL:
            fprintf(stderr, "\n    %18s(default=%s)\n", "",
                    gm_prop_get_bool(prop) ? "true" : "false");
            break;
        case GM_PROPERTY_STRING:
            str = gm_prop_get_string(prop);
            if (str)
                fprintf(stderr, "\n    %18s(default=%s)\n", "", str);
            else
                fprintf(stderr, "\n");
            break;
        default:
            fprintf(stderr, "\n");
            break;
        }
    }

    fprintf(stderr,
"\n"
"  -c, --continue             Continue attempting to process training runs,\n"
"                             even if there's a training run failure.\n"
"  -j, --threads=NUMBER       Number of threads to use (default: autodetect).\n"
"                             (can be overridden by per-run parameter)\n"
"  -s, --seed=NUMBER          Seed to use for RNG (default: 0).\n"
"                             (can be overridden by per-run parameter)\n"
"  -l, --log-file=FILE        File to write log message to.\n"
"      --log=FILE\n"
"\n"
"      --verbose              Verbose output.\n"
"  -h, --help                 Display this message.\n");
    exit(1);
}


static bool
boolean_string_value(struct gm_logger *log,
                     const char *boolean,
                     char **err)
{
    if (!boolean)
        return false;

    char lower[strlen(boolean) + 1];
    memcpy(lower, boolean, sizeof(lower));

    if (strcmp(lower, "true") == 0 ||
        strcmp(lower, "yes") == 0 ||
        strcmp(lower, "on") == 0 ||
        strcmp(lower, "1") == 0)
    {
        return true;
    } else if (strcmp(lower, "false") == 0 ||
               strcmp(lower, "no") == 0 ||
               strcmp(lower, "off") == 0 ||
               strcmp(lower, "0") == 0)
    {
        return false;
    } else {
        gm_throw(log, err, "Spurious boolean value of %s\n", boolean);
    }

    return true;
}

static bool
add_name_value_prop_to_json(struct gm_logger *log,
                            struct gm_ui_properties *props,
                            JSON_Value *props_object,
                            const char *name,
                            const char *value,
                            char **err)
{
    JSON_Object *json = json_object(props_object);

    for (int i = 0; i < props->n_properties; i++) {
        struct gm_ui_property *prop = &props->properties[i];
        char *end;
        int int_val;
        float float_val;
        bool bool_val;

        if (strcmp(prop->name, name) != 0)
            continue;

        switch (prop->type) {
        case GM_PROPERTY_INT:
            json_object_set_number(json, prop->name, atoi(value));
            int_val = strtol(optarg, &end, 0);
            if (*optarg == '\0' || *end != '\0') {
                gm_throw(log, err, "Failed to parse int value: '%s'", value);
                return false;
            }
            json_object_set_number(json, prop->name, int_val);
            break;
        case GM_PROPERTY_FLOAT:
            float_val = strtod(optarg, &end);
            if (*optarg == '\0' || *end != '\0') {
                gm_throw(log, err, "Failed to parse float value: '%s'", value);
                return false;
            }
            json_object_set_number(json, prop->name, float_val);
            break;
        case GM_PROPERTY_STRING:
            json_object_set_string(json, prop->name, value);
            break;
        case GM_PROPERTY_BOOL:
            bool_val = boolean_string_value(log, value, err);
            if (err && *err)
                return false;
            json_object_set_boolean(json, prop->name, bool_val);
            break;
        default:
            gm_throw(log, err,
                     "FIXME: Property type for %s can't be set on command line",
                     name);
            return false;
        }
        return true;
    }

    gm_throw(log, err, "Unknown property '%s'", name);
    return false;
}

int
main(int argc, char **argv)
{
    struct training_data _data;
    struct training_data *data = &_data;

    data->log_fp = stderr;
    data->log = gm_logger_new(logger_cb, data);
    gm_logger_set_abort_callback(data->log, logger_abort_cb, data);

    struct gm_rdt_context *ctx = gm_rdt_context_new(data->log);
    struct gm_ui_properties *ctx_props =
        gm_rdt_context_get_ui_properties(ctx);

#define VERBOSE_OPT    (CHAR_MAX + 1) // no short opt

    const char *short_options="q:p:v:d:cj:s:l:vh";
    const struct option long_options[] = {
        {"queue",        required_argument,  0, 'q'},
        {"parameter",    required_argument,  0, 'p'},
        {"property",     required_argument,  0, 'p'},
        {"value",        required_argument,  0, 'v'},
        {"data-dir",     required_argument,  0, 'd'},
        {"continue",     required_argument,  0, 'c'},
        {"threads",      required_argument,  0, 'j'},
        {"seed",         required_argument,  0, 's'},
        {"log",          required_argument,  0, 'l'},
        {"log-file",     required_argument,  0, 'l'},
        {"verbose",      no_argument,        0, VERBOSE_OPT},
        {"help",         no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

    JSON_Value *override_props_object = json_value_init_object();
    JSON_Value *work_queue = NULL;

    int opt;
    const char *prop_name = NULL;
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL))
           != -1)
    {
        if (prop_name && opt != 'v') {
            fprintf(stderr, "ERROR: Expected a -v,--value=VALUE after property name\n\n");
            exit(1);
        }

        char *err = NULL;
        struct stat sb;
        int len;

        switch (opt) {
        case 'd':
            data_dir_opt = strdup(optarg);
            break;
        case 'i':
            index_name_opt = strdup(optarg);
            break;
        case 'q':
            len = strlen(optarg);
            if (len > 5 && strcmp(&optarg[len-5], ".json") == 0) {
                if (stat(optarg, &sb) < 0) {
                    fprintf(stderr, "%s not found\n", optarg);
                    exit(1);
                }
                work_queue = json_parse_file_with_comments(optarg);
            } else {
                work_queue = json_parse_string_with_comments(optarg);
            }
            if (!work_queue || json_value_get_type(work_queue) != JSONArray) {
                fprintf(stderr, "Expected --queue,-q to be passed a JSON Array of jobs\n");
                exit(1);
            }
            break;
        case 'p':
            prop_name = optarg;
            break;
        case 'v':
            if (prop_name == NULL) {
                fprintf(stderr, "ERROR: Expected -p,--parameter=NAME before property value\n\n");
                exit(1);
            }
            if (!add_name_value_prop_to_json(data->log,
                                             ctx_props,
                                             override_props_object,
                                             prop_name,
                                             optarg,
                                             &err))
            {
                fprintf(stderr, "ERROR: Failed to override property '%s' with value '%s': %s\n\n",
                        prop_name,
                        optarg,
                        err);
                exit(1);
            }
            prop_name = NULL;
            break;
        case 'j':
            n_threads_opt = atoi(optarg);
            break;
        case 's':
            seed_opt = atoi(optarg);
            break;
        case 'l':
            data->log_fp = fopen(optarg, "w");
            break;
        case VERBOSE_OPT:
            verbose_opt = true;
            break;
        case 'c':
            continue_opt = true;
            break;
        case 'h':
            usage(ctx_props);
            break;
        default:
            usage(ctx_props);
            break;
        }
    }

    if (prop_name) {
        fprintf(stderr, "ERROR: Expected a -v,--value=VALUE after property name\n\n");
        exit(1);
    }

    if (argc - optind > 2) {
        fprintf(stderr, "ERROR: Too many arguments:\n\n");
        usage(ctx_props);
    }

    if (!work_queue) {
        if (argc - optind < 2) {
            fprintf(stderr, "ERROR: If no JSON work queue given then <index_name> and <results.json> required\n\n");
            exit(1);
        }
        index_name_opt = strdup(argv[optind]);
        out_file_opt = strdup(argv[optind + 1]);

        /* Create a stub work queue for consistency with using -q,--queue */
        work_queue = json_value_init_array();
        JSON_Value *run0 = json_value_init_object();
        json_array_append_value(json_array(work_queue), run0);
    } else {
        if (argc - optind > 0)
            index_name_opt = strdup(argv[optind]);
        if (argc - optind > 1)
            out_file_opt = strdup(argv[optind + 1]);
    }

    if (out_file_opt) {
        char *ext = strstr(out_file_opt, ".rdt");
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
            exit(1);
        }
    }

    /* The initial context is just created to so we can read default properties */
    gm_rdt_context_destroy(ctx);
    ctx_props = NULL;

    JSON_Array *work_array = json_array(work_queue);
    for (int i = 0; i < json_array_get_count(work_array); i++) {
        JSON_Value *run_props = json_array_get_value(work_array, i);

        gm_info(data->log, "Starting training run %d", i);

        ctx = gm_rdt_context_new(data->log);
        ctx_props = gm_rdt_context_get_ui_properties(ctx);

        /* Consistent with the --help documentation, configure the context
         * according to this precedence:
         *
         * - First, commandline options with lowest precedence
         * - Second, JSON run configurations
         * - Lastly, -p and -v options with highest precedence
         */

        if (data_dir_opt)
            gm_props_set_string(ctx_props, "data_dir", data_dir_opt);
        if (index_name_opt)
            gm_props_set_string(ctx_props, "index_name", index_name_opt);
        if (out_file_opt)
            gm_props_set_string(ctx_props, "out_file", out_file_opt);
        if (n_threads_opt)
            gm_props_set_int(ctx_props, "n_threads", n_threads_opt);
        if (seed_opt)
            gm_props_set_int(ctx_props, "seed", seed_opt);
        if (verbose_opt)
            gm_props_set_bool(ctx_props, "verbose", true);

        gm_props_from_json(data->log, ctx_props, run_props);
        gm_props_from_json(data->log, ctx_props, override_props_object);

        /* So that training run descriptions can avoid containing absolute
         * paths we treat the -d,--data-dir option specially and if the
         * "data_dir" property is a relative path we prepend/join with
         * the data_dir_opt path...
         */
        if (data_dir_opt) {
            const char *data_dir = gm_props_get_string(ctx_props, "data_dir");

            if (data_dir[0] != '/' && strcmp(data_dir_opt, data_dir) != 0) {
                char *joined_path = NULL;
                xasprintf(&joined_path, "%s/%s", data_dir_opt, data_dir);
                gm_props_set_string(ctx_props, "data_dir", joined_path);
                xfree(joined_path);
            }
        }

        char *err = NULL;
        if (!gm_rdt_context_train(ctx, &err)) {
            gm_error(data->log, "Failed to complete training run %d: %s\n", i, err);
            free(err);
            err = NULL;
            if (!continue_opt) {
                exit(1);
            }
        }

        gm_rdt_context_destroy(ctx);
        ctx_props = NULL;
    }

    json_value_free(work_queue);
    json_value_free(override_props_object);
    gm_logger_destroy(data->log);

    return 0;
}
