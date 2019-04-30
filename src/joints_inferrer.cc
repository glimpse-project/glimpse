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


#include <stdbool.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>

#include <vector>
#include <list>
#include <forward_list>
#include <thread>
#include <algorithm>
#include <cmath>

#include "glimpse_log.h"

#include "joints_inferrer.h"
#include "xalloc.h"
#include "jip.h"

#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.01f

#define ARRAY_LEN(ARRAY) (sizeof(ARRAY)/sizeof(ARRAY[0]))
#define CLAMP(x,min,max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))


struct joint_labels_entry {
    int n_labels;

    /* XXX: There's a runtime check that we don't have joints mapped to
     * more than two labels and this can easily be bumped if necessary.
     * Keeping this at the minimum size seems likely to help improve
     * locality considering the frequency of accessing this data within
     * some of the inner loops below.
     */
    uint8_t labels[2];
};

struct joints_inferrer
{
    struct gm_logger* log;

    int n_joints;
    std::vector<joint_labels_entry> map;

    int state_ref = 0;
};

struct joints_inferrer_state
{
    struct joints_inferrer *inferrer;

    std::vector<unsigned> cluster_id_runs;
    std::vector<unsigned> id_map;
    std::vector<std::vector<unsigned>> cluster_indices;

    std::vector<std::vector<Joint>> results;
};

float *
joints_inferrer_calc_pixel_weights(struct joints_inferrer_state *state,
                                   float *depth_image,
                                   float *pr_table,
                                   int width,
                                   int height,
                                   int n_labels,
                                   float *weights)
{
    struct joints_inferrer *inferrer = state->inferrer;
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

    gm_assert(inferrer->log, weights != NULL, "NULL weights destination buffer");

    for (int y = 0, weight_idx = 0, pixel_idx = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++, pixel_idx++)
        {
            float depth = (float)depth_image[pixel_idx];

            /* XXX: I think the idea here is to consider the surface area of
             * points, where points that are farther away really represent
             * a larger pysical surface than nearer points and so points
             * farther away will add more weight.
             *
             * Notably though this seems like a crude aproximation and maybe
             * we should actually consider the fov and camera resolution...
             */
            float depth_2 = depth * depth;

            for (int j = 0; j < n_joints; j++, weight_idx++)
            {
                float pr = 0.f;
                for (int n = 0; n < map[j].n_labels; n++)
                {
                    int label = (int)map[j].labels[n];
                    pr += pr_table[(pixel_idx * n_labels) + label];
                }
                weights[weight_idx] = pr * depth_2;
            }
        }
    }

    return weights;
}

// Clusters are first described as a sparse collection of per-line spans before
// we iterate through spans to associate them with cluster IDs. If spans are
// found to be vertically-adjacent to other spans then they should be mapped to
// the same cluster ID, but we can't simply edit the ID of spans we've already
// processed since we don't know what other spans might already be sharing a
// particular ID. The 'runs' array is a way of representing links or aliases
// for IDs temporarily and then at the end of clustering we go through and
// replace all IDs with the canonical 'root' ID that they alias/link to.
//
// So the merging of spans is done by maintaining an array of id ('runs') where
// normally runs[id] == id but when we need to track the merging of IDs then
// runs[id] can point back to an earlier id index. There may be multiple jumps
// like this due to multiple such merges so this function follows the
// indirections to find the 'root' id that will be the final effective id.
//
// Note: the indirections always point to lower indices, otherwise it would
// be possible to have loops.
//
static unsigned
find_id_root(std::vector<unsigned> &runs, unsigned index)
{
    unsigned idx = index;
    while (runs[idx] != idx)
        idx = runs[idx];

    // In case there were multiple indirections then updating the index
    // will reduce all further lookups to one indirection at most
    //runs[index] = idx; (Didn't improve performance)

    return idx;
}

InferredJoints*
joints_inferrer_infer_fast(struct joints_inferrer_state *state,
                           struct gm_intrinsics *intrinsics,
                           int cluster_width,
                           int cluster_height,
                           int cluster_x0,
                           int cluster_y0,
                           float *cluster_depth_image,
                           float *cluster_label_probs,
                           float *cluster_weights,
                           int n_labels,
                           JIParam *params)
{
    struct joints_inferrer *inferrer = state->inferrer;
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

    // Variables for reprojection of 2d point + depth
    float fx = intrinsics->fx;
    float fy = intrinsics->fy;
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    float cx = intrinsics->cx;
    float cy = intrinsics->cy;

    // Plan: For each scan-line, scan along and record clusters on 1 dimension.
    //       For each scanline segment, check to see if it intersects with any
    //       segment on neighbouring scanlines, then join the lists together.
    //       Finally, we should have a list of lists of clusters, which we
    //       can then calculate confidence for, then when we have the highest
    //       confidence for each label, we can project the points and calculate
    //       the center-point.
    //
    //       TODO: Let this take a distance so that clusters don't need to be
    //             perfectly contiguous?
    //       TODO: Figure out a way to divide clusters that are only loosely
    //             connected?
    struct cluster_span {
        int left;
        int right;
        int id;
    };

    // The 2D clusters for each joint may not be mutually exclusive so
    // for each joint we build a separate (sparse) representation of
    // clusters.
    //
    // The clusters themselves are first described as a vector of 1D spans
    // for each line in the image.
    //
    // Afterwards we try and merge spans that are connected vertically
    // on adjacent lines.
    //
    std::vector<std::vector<cluster_span>> lines[n_joints];
    for (int j = 0; j < n_joints; j++)
    {
        lines[j].resize(cluster_height);
    }

    // Collect clusters across scanlines
    for (int y = 0; y < cluster_height; ++y)
    {
        for (int x = 0; x < cluster_width; ++x)
        {
            for (int j = 0; j < n_joints; ++j)
            {
                bool threshold_passed = false;
                for (int n = 0; n < map[j].n_labels; ++n)
                {
                    int label = (int)map[j].labels[n];
                    float label_pr = cluster_label_probs[(y * cluster_width + x) * n_labels + label];
                    if (label_pr >= params[j].threshold)
                    {
                        threshold_passed = true;
                        break;
                    }
                }

                if (threshold_passed)
                {
                    // Check to see if this pixel can be added to an existing
                    // cluster.
                    if (lines[j][y].size() &&
                        lines[j][y].back().right == (x - 1))
                    {
                        lines[j][y].back().right = x;
                    }
                    else
                    {
                        lines[j][y].push_back({x, x, -1});
                    }
                }
            }
        }
    }

    std::vector<unsigned> &cluster_id_runs = state->cluster_id_runs;
    std::vector<unsigned> &id_map = state->id_map;
    std::vector<std::vector<unsigned>> &cluster_indices = state->cluster_indices;

    std::vector<std::vector<Joint>> &results = state->results;
    results.resize(n_joints);

    // Now iteratively connect the scanline clusters
    for (int j = 0; j < n_joints; j++)
    {
        cluster_id_runs.resize(0);
        id_map.resize(0);
        cluster_indices.resize(0);
        results[j].resize(0);

        for (auto &span : lines[j][0])
        {
            span.id = cluster_id_runs.size();
            cluster_id_runs.push_back(span.id);
        }

        for (int y = 1; y < cluster_height; y++)
        {
            for (auto &span : lines[j][y])
            {
                for (auto &span_above : lines[j][y - 1])
                {
                    if ((span.left <= span_above.right) &&
                        (span.right >= span_above.left))
                    {
                        if (span.id == -1) {
                            span.id = span_above.id;
                        } else {
                            unsigned root_self = find_id_root(cluster_id_runs, span.id);
                            unsigned root_above = find_id_root(cluster_id_runs, span_above.id);
                            if (root_self < root_above) {
                                cluster_id_runs[root_above] = root_self;
                            } else {
                                cluster_id_runs[root_self] = root_above;
                            }
                        }
                    }
                }
                if (span.id == -1) {
                    span.id = cluster_id_runs.size();
                    cluster_id_runs.push_back(span.id);
                }
            }
        }

        id_map.resize(cluster_id_runs.size());
        unsigned max_id = 0;
        for (unsigned i = 0; i < cluster_id_runs.size(); i++)
        {
            // if it is its own root -> new region
            if (cluster_id_runs[i] == i)
                id_map[i] = max_id++;
            else // assign this sub-segment to the region (root) it belongs
                id_map[i] = id_map[find_id_root(cluster_id_runs, i)];
        }

        cluster_indices.resize(max_id);
        for (int y = 0; y < cluster_height; y++)
        {
            for (auto &span : lines[j][y])
            {
                span.id = id_map[span.id];
                for (int x = span.left; x <= span.right; x++) {
                    int idx = cluster_width * y + x;
                    cluster_indices[span.id].push_back(idx);
                }
            }
        }

        for (auto &cluster : cluster_indices) {
            Joint joint;
            joint.confidence = 0.f;

            int n_points = cluster.size();
            int x_sum = 0;
            int y_sum = 0;

            // Calculate the center-point and confidence of the cluster
            for (unsigned i : cluster) {
                int x = i % cluster_width;
                int y = i / cluster_width;

                x_sum += x;
                y_sum += y;
                joint.confidence += cluster_weights[i * n_joints + j];
            }

            int x = roundf(x_sum / (float)n_points);
            int y = roundf(y_sum / (float)n_points);

            // Find the nearest point in the cluster - the coordinates above
            // aren't guaranteed to be in the cluster (though they more often
            // than not are, they aren't frequently enough that we can't rely
            // on that).
            float min_squared_sdist = std::numeric_limits<float>::max();
            int nx = x, ny = y;

            for (unsigned i : cluster) {
                int sx = i % cluster_width;
                int sy = i / cluster_width;
                float dx = x - sx;
                float dy = y - sy;
                float squared_sdist = dx * dx + dy * dy;
                if (squared_sdist < min_squared_sdist) {
                    nx = sx;
                    ny = sy;
                    min_squared_sdist = squared_sdist;
                }
                if (nx == x && ny == y) {
                    break;
                }
            }

            // Reproject and offset point
            float depth = (float)cluster_depth_image[ny * cluster_width + nx];

            joint.x = ((nx + cluster_x0) - cx) * depth * inv_fx;
            // NB: The coordinate space for joints has Y+ extending upwards...
            joint.y = -(((ny + cluster_y0) - cy) * depth * inv_fy);
            joint.z = depth + params[j].offset;

            results[j].push_back(joint);
        }

        std::sort(results[j].begin(), results[j].end(),
                  [](Joint &a, Joint &b){ return a.confidence > b.confidence; });
    }


    /* TODO: remove this linked list fiddling, and return a packed array-based
     * representation of the resuts...
     */

    InferredJoints *ret = (InferredJoints*)xcalloc(sizeof(InferredJoints) +
                                                   sizeof(LList*) * n_joints,
                                                   1);
    ret->n_joints = n_joints;

    for (int j = 0; j < n_joints; j++) {
        if (!results[j].size())
            continue;

        Joint *copy = (Joint *)xmalloc(sizeof(Joint));
        memcpy(copy, &results[j][0], sizeof(Joint));

        LList *first = llist_new(copy);
        LList *last = first;
        for (int i = 1; i < (int)results[j].size(); i++)
        {
            copy = (Joint *)xmalloc(sizeof(Joint));
            memcpy(copy, &results[j][i], sizeof(Joint));
            last = llist_insert_after(last, llist_new(copy));
        }
        ret->joints[j] = first;
    }

    return ret;
}

static int
compare_joints(LList *a, LList *b, void *userdata)
{
    Joint* ja = (Joint*)a->data;
    Joint* jb = (Joint*)b->data;
    return ja->confidence - jb->confidence;
}

InferredJoints*
joints_inferrer_infer(struct joints_inferrer_state *state,
                      struct gm_intrinsics *intrinsics,
                      int cluster_width,
                      int cluster_height,
                      int cluster_x0,
                      int cluster_y0,
                      float *cluster_depth_image,
                      float *cluster_label_probs,
                      float *cluster_weights,
                      float bg_depth,
                      int n_labels,
                      JIParam* params)
{
    struct joints_inferrer *inferrer = state->inferrer;
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

    // Use mean-shift to find the inferred joint positions, set them back into
    // the body using the given offset, and return the results
    int *n_pixels = (int *)xcalloc(n_joints, sizeof(int));
    size_t points_size = n_joints * cluster_width * cluster_height * 3 * sizeof(float);
    float *points = (float *)xmalloc(points_size);
    float *density = (float *)xmalloc(points_size);

    // Variables for reprojection of 2d point + depth
    float fx = intrinsics->fx;
    float fy = intrinsics->fy;
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    float cx = intrinsics->cx;
    float cy = intrinsics->cy;

    float root_2pi = sqrtf(2 * M_PI);

    int too_many_pixels = (cluster_width * cluster_height) / 2;

    // Gather pixels above the given threshold
    for (int y = 0, idx = 0; y < cluster_height; y++)
    {
        for (int x = 0; x < cluster_width; x++, idx++)
        {
            float depth = (float)cluster_depth_image[idx];
            if (!std::isnormal(depth) || depth >= bg_depth)
            {
                continue;
            }

            for (int j = 0; j < n_joints; j++)
            {
                float threshold = params[j].threshold;
                int joint_idx = j * cluster_width * cluster_height;

                for (int n = 0; n < map[j].n_labels; n++)
                {
                    int label = (int)map[j].labels[n];
                    float label_pr = cluster_label_probs[(idx * n_labels) + label];
                    if (label_pr >= threshold)
                    {
                        // Reproject point
                        points[(joint_idx + n_pixels[j]) * 3] =
                            ((x + cluster_x0) - cx) * depth * inv_fx;
                        points[(joint_idx + n_pixels[j]) * 3 + 1] =
                            -(((y + cluster_y0) - cy) * depth * inv_fy);
                        points[(joint_idx + n_pixels[j]) * 3 + 2] =
                            depth;

                        // Store pixel weight (density)
                        density[joint_idx + n_pixels[j]] =
                            cluster_weights[(idx * n_joints) + j];

                        n_pixels[j]++;
                        break;
                    }
                }
            }
        }
    }

    InferredJoints *result = (InferredJoints*)xcalloc(sizeof(InferredJoints) +
                                                      sizeof(LList*) * n_joints,
                                                      1);
    result->n_joints = n_joints;

    // Means shift to find joint modes
    for (int j = 0; j < n_joints; j++)
    {
        if (n_pixels[j] == 0 || n_pixels[j] > too_many_pixels)
        {
            continue;
        }

        float bandwidth = params[j].bandwidth;
        float offset = params[j].offset;

        int joint_idx = j * cluster_width * cluster_height;
        for (int s = 0; s < N_SHIFTS; s++)
        {
            float new_points[n_pixels[j] * 3];
            bool moved = false;
            for (int p = 0; p < n_pixels[j]; p++)
            {
                float* x = &points[(joint_idx + p) * 3];
                float* nx = &new_points[p * 3];
                float numerator[3] = { 0.f, };
                float denominator = 0.f;
                for (int n = 0; n < n_pixels[j]; n++)
                {
                    float* xi = &points[(joint_idx + n) * 3];
                    float distance = sqrtf(pow(x[0] - xi[0], 2.f) +
                                           pow(x[1] - xi[1], 2.f) +
                                           pow(x[2] - xi[2], 2.f));

                    // Weighted gaussian kernel
                    float weight = density[joint_idx + n] *
                        (1.f / (bandwidth * root_2pi)) *
                        expf(-0.5f * pow(distance / bandwidth, 2.f));

                    numerator[0] += weight * xi[0];
                    numerator[1] += weight * xi[1];
                    numerator[2] += weight * xi[2];

                    denominator += weight;
                }

                nx[0] = numerator[0] / denominator;
                nx[1] = numerator[1] / denominator;
                nx[2] = numerator[2] / denominator;

                if (!moved &&
                    (fabs(nx[0] - x[0]) >= SHIFT_THRESHOLD ||
                     fabs(nx[1] - x[1]) >= SHIFT_THRESHOLD ||
                     fabs(nx[2] - x[2]) >= SHIFT_THRESHOLD))
                {
                    moved = true;
                }
            }

            memcpy((void*)&points[joint_idx * 3], (void*)new_points,
                   n_pixels[j] * 3 * sizeof(float));

            if (!moved || s == N_SHIFTS - 1)
            {
                // Calculate the confidence of all modes found
                float* last_point = &points[joint_idx * 3];
                Joint* joint = (Joint*)xmalloc(sizeof(Joint));
                joint->x = last_point[0];
                joint->y = last_point[1];
                joint->z = last_point[2] + offset;
                joint->confidence = 0;
                result->joints[j] = llist_new(joint);

                //int unique_points = 1;

                for (int p = 0; p < n_pixels[j]; p++)
                {
                    float* point = &points[(joint_idx + p) * 3];
                    if (fabs(point[0]-last_point[0]) >= SHIFT_THRESHOLD ||
                        fabs(point[1]-last_point[1]) >= SHIFT_THRESHOLD ||
                        fabs(point[2]-last_point[2]) >= SHIFT_THRESHOLD)
                    {
                        //unique_points++;
                        last_point = point;
                        joint = (Joint*)xmalloc(sizeof(Joint));
                        joint->x = last_point[0];
                        joint->y = last_point[1];
                        joint->z = last_point[2] + offset;
                        joint->confidence = 0;
                        result->joints[j] = llist_insert_before(result->joints[j],
                                                                llist_new(joint));
                    }
                    joint->confidence += density[joint_idx + p];
                }

                llist_sort(result->joints[j], compare_joints, NULL);

                break;
            }
        }
    }

    xfree(density);
    xfree(points);
    xfree(n_pixels);

    return result;
}

void
joints_inferrer_state_free_joints(struct joints_inferrer_state *state,
                                  InferredJoints *joints)
{
    for (int i = 0; i < joints->n_joints; i++) {
        llist_free(joints->joints[i], llist_free_cb, NULL);
    }
    xfree(joints);
}

struct joints_inferrer *
joints_inferrer_new(struct gm_logger *log,
                    JSON_Value *joint_map,
                    char **err)
{
    struct joints_inferrer *inferrer = new joints_inferrer();

    inferrer->log = log;
    inferrer->state_ref = 0;

    int n_joints = json_array_get_count(json_array(joint_map));
    inferrer->n_joints = n_joints;

    std::vector<joint_labels_entry> &map = inferrer->map;
    map.resize(n_joints);

    for (int i = 0; i < n_joints; i++) {
        JSON_Object *entry = json_array_get_object(json_array(joint_map), i);
        JSON_Array *labels = json_object_get_array(entry, "labels");
        int n_labels = json_array_get_count(labels);

        if (n_labels > (int)ARRAY_LEN(map[0].labels)) {
            gm_throw(log, err, "Didn't expect joint to be mapped to > 2 labels\n");
            delete inferrer;
            return NULL;
        }

        map[i].n_labels = n_labels;
        for (int n = 0; n < n_labels; n++) {
            map[i].labels[n] = json_array_get_number(labels, n);
        }
    }

    return inferrer;
}

void
joints_inferrer_destroy(struct joints_inferrer *inferrer)
{
    gm_assert(inferrer->log, inferrer->state_ref == 0,
              "Can't destroying joints inferrer before destroying all associated joints inferrer state");

    delete inferrer;
}

struct joints_inferrer_state *
joints_inferrer_state_new(struct joints_inferrer *inferrer)
{
    struct joints_inferrer_state *state = new joints_inferrer_state();
    state->inferrer = inferrer;

    inferrer->state_ref++;

    return state;
}

void
joints_inferrer_state_destroy(struct joints_inferrer_state *state)
{
    state->inferrer->state_ref--;
    delete state;
}

