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

#pragma once

#include <vector>

#include "joints_inferrer.h"

struct joints_inferrer_state
{
    struct joints_inferrer *inferrer;

    int width;
    int height;

    std::vector<unsigned> cluster_id_runs;
    std::vector<unsigned> id_map;
    std::vector<std::vector<unsigned>> cluster_indices;

    std::vector<std::vector<Joint>> results;

    // Normally we infer clusters for one joint and then discard
    // the indices when moving on to the next joint. For debugging
    // though (debug=true when calling _infer api) then we copy
    // the per-joint indices here before moving to the next joint
    // so that it's possible to visualize the clusters.
    //
    // indexed by joint index, then, cluster id...
    std::vector<std::vector<std::vector<unsigned>>> debug_joint_clusters;
    bool debug_joint_clusters_valid;
};


