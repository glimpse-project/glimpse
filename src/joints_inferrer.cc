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
#include <math.h>
#include <pthread.h>

#include <vector>
#include <list>
#include <forward_list>
#include <thread>
#include <cmath>

#include "glimpse_log.h"

#include "joints_inferrer.h"
#include "xalloc.h"
#include "jip.h"

#define N_SHIFTS 5
#define SHIFT_THRESHOLD 0.01f

#define ARRAY_LEN(ARRAY) (sizeof(ARRAY)/sizeof(ARRAY[0]))


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
};


float*
joints_inferrer_calc_pixel_weights(struct joints_inferrer *inferrer,
                                   float* depth_image,
                                   float* pr_table,
                                   int width, int height,
                                   int n_labels,
                                   float* weights)
{
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

    gm_assert(inferrer->log, weights != NULL, "NULL weights destination buffer");

    for (int y = 0, weight_idx = 0, pixel_idx = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++, pixel_idx++)
        {
            float depth = (float)depth_image[pixel_idx];
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

static int
compare_joints(LList* a, LList* b, void* userdata)
{
    Joint* ja = (Joint*)a->data;
    Joint* jb = (Joint*)b->data;
    return ja->confidence - jb->confidence;
}

InferredJoints*
joints_inferrer_infer_fast(struct joints_inferrer* inferrer,
                           float* depth_image,
                           float* pr_table,
                           float* weights,
                           int width, int height, int n_labels,
                           float vfov,
                           JIParam* params)
{
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

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
    typedef struct {
        int y;
        int left;
        int right;
    } ScanlineSegment;

    typedef std::forward_list<ScanlineSegment> Cluster;
    typedef std::move_iterator<typename Cluster::iterator> ClusterMoveIterator;

    typename std::list<Cluster> clusters[n_joints];

    // Collect clusters across scanlines
    ScanlineSegment* last_segment[n_joints];
    for (int y = 0; y < height; ++y)
    {
        memset(last_segment, 0, sizeof(ScanlineSegment*) * n_joints);
        for (int x = 0; x < width; ++x)
        {
            for (int j = 0; j < n_joints; ++j)
            {
                bool threshold_passed = false;
                for (int n = 0; n < map[j].n_labels; ++n)
                {
                    int label = (int)map[j].labels[n];
                    float label_pr = pr_table[(y * width + x) * n_labels + label];
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
                    if (last_segment[j])
                    {
                        last_segment[j]->right = x;
                    }
                    else
                    {
                        clusters[j].emplace_front();
                        clusters[j].front().push_front({ y, x, x });
                        last_segment[j] = &clusters[j].front().front();
                    }
                }
                else
                {
                    last_segment[j] = nullptr;
                }
            }
        }
    }

    // Now iteratively connect the scanline clusters
    for (int j = 0; j < n_joints; j++)
    {
        for (typename std::list<Cluster>::iterator it = clusters[j].begin();
             it != clusters[j].end(); ++it)
        {
            Cluster& parent = *it;
            for (typename std::list<Cluster>::iterator it2 = clusters[j].begin();
                 it2 != clusters[j].end();)
            {
                Cluster& candidate = *it2;
                if (it == it2)
                {
                    ++it2;
                    continue;
                }

                // If this scanline segment connects to the cluster being
                // checked, remove it from the cluster list, add it to the
                // checked cluster and break out.
                bool local_change = false;
                for (typename Cluster::iterator p_it = parent.begin();
                     p_it != parent.end() && !local_change; ++p_it)
                {
                    ScanlineSegment& p_segment = *p_it;
                    for (typename Cluster::iterator c_it = candidate.begin();
                         c_it != candidate.end(); ++c_it)
                    {
                        ScanlineSegment& c_segment = *c_it;
                        // Check if these two scanline cluster segments touch
                        if ((abs(c_segment.y - p_segment.y) == 1) &&
                            (c_segment.left <= p_segment.right) &&
                            (c_segment.right >= p_segment.left))
                        {
                            parent.insert_after(
                                parent.before_begin(),
                                ClusterMoveIterator(candidate.begin()),
                                ClusterMoveIterator(candidate.end()));
                            it2 = clusters[j].erase(it2);
                            local_change = true;
                            break;
                        }
                    }
                }

                if (!local_change)
                {
                    ++it2;
                }
            }
        }
    }

    // clusters now contains the boundaries per scanline segment of each cluster
    // of joint labels, which we can now use to calculate the highest confidence
    // cluster and the projected cluster centroid.

    // Variables for reprojection of 2d point + depth
    float half_width = width / 2.f;
    float half_height = height / 2.f;
    float aspect = half_width / half_height;

    float vfov_rad = vfov * M_PI / 180.f;
    float tan_half_vfov = tanf(vfov_rad / 2.f);
    float tan_half_hfov = tan_half_vfov * aspect;
    //float hfov = atanf(tan_half_hfov) * 2.f;

    //float root_2pi = sqrtf(2.f * M_PI);

    // Allocate/clear joints structure
    InferredJoints* result = (InferredJoints*)xmalloc(sizeof(InferredJoints));
    result->n_joints = n_joints;
    result->joints = (LList**)xcalloc(n_joints, sizeof(LList*));

    for (int j = 0; j < n_joints; j++)
    {
        for (typename std::list<Cluster>::iterator it =
             clusters[j].begin(); it != clusters[j].end(); ++it)
        {
            Cluster& cluster = *it;
            Joint* joint = (Joint*)xmalloc(sizeof(Joint));

            // Calculate the center-point and confidence of the cluster
            int n_points = 0;
            int x = 0;
            int y = 0;
            joint->confidence = 0.f;
            for (typename Cluster::iterator s_it = cluster.begin();
                 s_it != cluster.end(); ++s_it)
            {
                ScanlineSegment& segment = *s_it;
                int idx = segment.y * width;
                for (int i = segment.left; i <= segment.right; i++, n_points++)
                {
                    x += i;
                    y += segment.y;
                    joint->confidence += weights[(idx + i) * n_joints + j];
                }
            }

            x = (int)roundf(x / (float)n_points);
            y = (int)roundf(y / (float)n_points);

            // Reproject and offset point
            float s = (x / half_width) - 1.f;
            // NB: The coordinate space for joints has Y+ extending upwards...
            float t = -((y / half_height) - 1.f);
            float depth = (float)depth_image[y * width + x];
            joint->x = (tan_half_hfov * depth) * s;
            joint->y = (tan_half_vfov * depth) * t;
            joint->z = depth + params[j].offset;

            // Add the joint to the list
            result->joints[j] = llist_insert_before(result->joints[j],
                                                    llist_new(joint));
        }

        llist_sort(result->joints[j], compare_joints, NULL);
    }

    return result;
}

InferredJoints*
joints_inferrer_infer(struct joints_inferrer* inferrer,
                      float* depth_image,
                      float* pr_table,
                      float* weights,
                      int width,
                      int height,
                      float bg_depth,
                      int n_labels,
                      float vfov,
                      JIParam* params)
{
    int n_joints = inferrer->n_joints;
    std::vector<joint_labels_entry> &map = inferrer->map;

    // Use mean-shift to find the inferred joint positions, set them back into
    // the body using the given offset, and return the results
    int* n_pixels = (int*)xcalloc(n_joints, sizeof(int));
    size_t points_size = n_joints * width * height * 3 * sizeof(float);
    float* points = (float*)xmalloc(points_size);
    float* density = (float*)xmalloc(points_size);

    // Variables for reprojection of 2d point + depth
    float half_width = width / 2.f;
    float half_height = height / 2.f;
    float aspect = half_width / half_height;

    float vfov_rad = vfov * M_PI / 180.f;
    float tan_half_vfov = tanf(vfov_rad / 2.f);
    float tan_half_hfov = tan_half_vfov * aspect;
    //float hfov = atanf(tan_half_hfov) * 2;

    float root_2pi = sqrtf(2 * M_PI);

    int too_many_pixels = (width * height) / 2;

    // Gather pixels above the given threshold
    for (int y = 0, idx = 0; y < height; y++)
    {
        float t = -((y / half_height) - 1.f);
        for (int x = 0; x < width; x++, idx++)
        {
            float s = (x / half_width) - 1.f;
            float depth = (float)depth_image[idx];
            if (!std::isnormal(depth) || depth >= bg_depth)
            {
                continue;
            }

            for (int j = 0; j < n_joints; j++)
            {
                float threshold = params[j].threshold;
                int joint_idx = j * width * height;

                for (int n = 0; n < map[j].n_labels; n++)
                {
                    int label = (int)map[j].labels[n];
                    float label_pr = pr_table[(idx * n_labels) + label];
                    if (label_pr >= threshold)
                    {
                        // Reproject point
                        points[(joint_idx + n_pixels[j]) * 3] =
                            (tan_half_hfov * depth) * s;
                        points[(joint_idx + n_pixels[j]) * 3 + 1] =
                            (tan_half_vfov * depth) * t;
                        points[(joint_idx + n_pixels[j]) * 3 + 2] =
                            depth;

                        // Store pixel weight (density)
                        density[joint_idx + n_pixels[j]] =
                            weights[(idx * n_joints) + j];

                        n_pixels[j]++;
                        break;
                    }
                }
            }
        }
    }

    InferredJoints* result = (InferredJoints*)xmalloc(sizeof(InferredJoints));
    result->n_joints = n_joints;
    result->joints = (LList**)xcalloc(n_joints, sizeof(LList*));

    // Means shift to find joint modes
    for (int j = 0; j < n_joints; j++)
    {
        if (n_pixels[j] == 0 || n_pixels[j] > too_many_pixels)
        {
            continue;
        }

        float bandwidth = params[j].bandwidth;
        float offset = params[j].offset;

        int joint_idx = j * width * height;
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
joints_inferrer_free_joints(struct joints_inferrer* inferrer,
                            InferredJoints* joints)
{
    for (int i = 0; i < joints->n_joints; i++) {
        llist_free(joints->joints[i], llist_free_cb, NULL);
    }
    xfree(joints->joints);
    xfree(joints);
}

struct joints_inferrer*
joints_inferrer_new(struct gm_logger* log,
                    JSON_Value* joint_map,
                    char** err)
{
    struct joints_inferrer* inferrer = new joints_inferrer();

    inferrer->log = log;

    int n_joints = json_array_get_count(json_array(joint_map));
    inferrer->n_joints = n_joints;

    std::vector<joint_labels_entry> &map = inferrer->map;
    map.resize(n_joints);

    for (int i = 0; i < n_joints; i++) {
        JSON_Object* entry = json_array_get_object(json_array(joint_map), i);
        JSON_Array* labels = json_object_get_array(entry, "labels");
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
joints_inferrer_destroy(struct joints_inferrer* inferrer)
{
    delete inferrer;
}
