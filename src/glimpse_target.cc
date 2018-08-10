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
                         const char *index_asset_name,
                         char **err)
{
    // Load the JSON file index
    struct gm_asset *index_asset =
        gm_asset_open(log, index_asset_name, GM_ASSET_MODE_BUFFER, err);
    if (!index_asset) {
        return NULL;
    }

    const char *buf = (const char *)gm_asset_get_buffer(index_asset);
    if (!buf) {
        gm_throw(log, err, "Error retrieving buffer from asset '%s'",
                 index_asset_name);
        gm_asset_close(index_asset);
        return NULL;
    }

    struct gm_target *self = gm_target_new(ctx, log);

    char file[1024];
    char *end_of_base = (char *)strrchr(index_asset_name, '/');
    if (end_of_base) {
        strncpy(file, index_asset_name, end_of_base - index_asset_name);
        end_of_base = &file[end_of_base - index_asset_name] + 1;
        *(end_of_base-1) = '/';
    } else {
        end_of_base = file;
    }

    const char *end = buf;
    while ((end = strchr(buf, '\n'))) {
        if (end - buf > 1) {
            strncpy(end_of_base, buf, end - buf);
            end_of_base[end - buf] = '\0';
            gm_debug(log, "XXX Trying to open skeleton '%s'", file);

            struct gm_skeleton *skeleton = gm_skeleton_new_from_json(ctx, file);
            if (skeleton) {
                self->frames.push_back(skeleton);
            } else {
                gm_warn(log, "Error opening skeleton asset '%s'", file);
            }
        }
        buf = end + 1;
    }

    gm_asset_close(index_asset);

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

const struct gm_skeleton *
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

float
gm_target_get_cumulative_error(struct gm_target *target,
                               const struct gm_skeleton *skeleton)
{
    float err = 0.f;
    int n_bones = gm_skeleton_get_n_bones(skeleton);
    for (int i = 0; i < n_bones; ++i) {
        err += gm_target_get_error(target,
                                   gm_skeleton_get_bone(skeleton, i));
    }
    return err / (float)n_bones;
}

float
gm_target_get_error(struct gm_target *target,
                    const struct gm_bone *bone)
{
    if (target->frame >= target->frames.size()) {
        return 1.f;
    }

    const struct gm_bone *ref_bone =
        gm_skeleton_find_bone(target->frames[target->frame],
                              gm_bone_get_head(bone),
                              gm_bone_get_tail(bone));
    if (!ref_bone) {
        return 1.f;
    }

    float xyzw[4];
    gm_bone_get_angle(bone, xyzw);
    glm::quat bone_angle(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    gm_bone_get_angle(ref_bone, xyzw);
    glm::quat ref_bone_angle(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);

    float angle = glm::degrees(glm::angle(
        glm::normalize(bone_angle) *
        glm::inverse(glm::normalize(ref_bone_angle))));
    while (angle > 180.f) angle -= 360.f;

    return std::min(90.f, fabsf(angle)) / 90.f;
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
