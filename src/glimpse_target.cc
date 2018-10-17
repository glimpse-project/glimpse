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

#include <string.h>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include "parson.h"

#include "glimpse_target.h"
#include "glimpse_assets.h"

struct gm_target {
    struct gm_context *ctx;
    struct gm_logger *log;

    std::vector<struct gm_skeleton *> frames;
    int frame;
};

struct gm_target *
gm_target_new(struct gm_context *ctx,
              struct gm_logger *log)
{
    struct gm_target *self = new struct gm_target;
    self->ctx = ctx;
    self->log = log;
    self->frame = 0;
    return self;
}

struct gm_target *
gm_target_new_from_index(struct gm_context *ctx,
                         struct gm_logger *log,
                         const char *target_sequence_name,
                         char **err)
{
    // Load the JSON file index
    struct gm_asset *target_sequence_asset =
        gm_asset_open(log, target_sequence_name, GM_ASSET_MODE_BUFFER, err);
    if (!target_sequence_asset) {
        return NULL;
    }

    const char *buf = (const char *)gm_asset_get_buffer(target_sequence_asset);
    if (!buf) {
        gm_throw(log, err, "Error retrieving buffer from asset '%s'",
                 target_sequence_name);
        gm_asset_close(target_sequence_asset);
        return NULL;
    }
    int len = gm_asset_get_length(target_sequence_asset);

    /* unfortunately parson doesn't support parsing from a buffer with
     * a given length and expects a NUL terminated string...
     */
    char *js_string = (char *)xmalloc(len + 1);

    memcpy(js_string, buf, len);
    js_string[len] = '\0';

    JSON_Value *target_sequence_value = json_parse_string(js_string);

    xfree(js_string);
    js_string = NULL;
    gm_asset_close(target_sequence_asset);
    target_sequence_asset = NULL;

    if (!target_sequence_value) {
        gm_throw(log, err, "Failed to parse target sequence %s",
                 target_sequence_name);
        return NULL;
    }

    struct gm_target *self = gm_target_new(ctx, log);

    int context_n_joints = gm_context_get_n_joints(ctx);

    JSON_Array *frames = json_object_get_array(json_object(target_sequence_value),
                                               "frames");
    int n_frames = json_array_get_count(frames);
    for (int i = 0; i < n_frames; i++) {
        JSON_Object *frame = json_array_get_object(frames, i);
        JSON_Array *joints_js = json_object_get_array(frame, "joints");

        int n_joints = json_array_get_count(joints_js);
        if (n_joints != context_n_joints) {
            gm_throw(log, err, "Target sequence %s frame %d had %d joints, but expected %d joints",
                     target_sequence_name, i, n_joints, context_n_joints);
            gm_target_free(self);
            json_value_free(target_sequence_value);
            return NULL;
        }

        struct gm_joint joints[n_joints];
        for (int j = 0; j < n_joints; j++) {
            JSON_Object *joint = json_array_get_object(joints_js, j);
            joints[j].valid = true;
            joints[j].x = json_object_get_number(joint, "x");
            joints[j].y = json_object_get_number(joint, "y");
            joints[j].z = json_object_get_number(joint, "z");
        }

        struct gm_skeleton *skeleton = gm_skeleton_new(ctx, joints);
        self->frames.push_back(skeleton);
    }

    json_value_free(target_sequence_value);
    target_sequence_value = NULL;

    return self;
}

void
gm_target_insert_frame(struct gm_target *target,
                       struct gm_skeleton *skeleton,
                       unsigned int index)
{
    target->frames.insert(target->frames.begin() + index, skeleton);
}

void
gm_target_remove_frame(struct gm_target *target,
                       unsigned int index)
{
    if (index >= target->frames.size()) {
        gm_warn(target->log, "Tried to remove non-existent frame %u", index);
        return;
    }

    gm_skeleton_free(target->frames[index]);
    target->frames.erase(target->frames.begin() + index);
}

unsigned int
gm_target_get_n_frames(struct gm_target *target)
{
    return (unsigned int)target->frames.size();
}

struct gm_skeleton *
gm_target_get_skeleton(struct gm_target *target)
{
    if (target->frame >= target->frames.size()) {
        return NULL;
    }

    return target->frames[target->frame];
}

unsigned int
gm_target_get_frame(struct gm_target *target)
{
    return target->frame;
}

void
gm_target_set_frame(struct gm_target *target, unsigned int frame)
{
    if (frame < target->frames.size()) {
        target->frame = frame;
    }
}

void
gm_target_free(struct gm_target *target)
{
    for (std::vector<struct gm_skeleton *>::iterator it = target->frames.begin();
         it != target->frames.end(); ++it) {
        gm_skeleton_free(*it);
    }
    delete target;
}
