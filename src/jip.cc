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

#include <sys/types.h>
#include <sys/stat.h>

#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cstddef>
#include <stdio.h>

#include "jip.h"
#include "parson.h"
#include "xalloc.h"

JIParams *
jip_load_from_json(JSON_Value *root)
{
    JIParams* jip = (JIParams*)xcalloc(1, sizeof(JIParams));

    jip->header.tag[0] = 'J';
    jip->header.tag[1] = 'I';
    jip->header.tag[2] = 'P';

    jip->header.version = 0;

    jip->header.n_joints = json_object_get_number(json_object(root), "n_joints");

    JSON_Array *params = json_object_get_array(json_object(root), "params");
    int len = json_array_get_count(params);
    if (len != jip->header.n_joints)
    {
        fprintf(stderr, "Inconsistency between \"n_joints\" and length of \"params\" array\n");
        free(jip);
        return NULL;
    }

    jip->joint_params = (JIParam*)xmalloc(jip->header.n_joints * sizeof(JIParam));
    for (int i = 0; i < len; i++)
    {
        JSON_Object *param = json_array_get_object(params, i);

        jip->joint_params[i].bandwidth = json_object_get_number(param, "bandwidth");
        jip->joint_params[i].threshold = json_object_get_number(param, "threshold");
        jip->joint_params[i].offset = json_object_get_number(param, "offset");
    }

    return jip;
}

JIParams*
jip_load_from_file(const char* filename)
{
    const char* ext;

    if ((ext = strstr(filename, ".json")) && ext[5] == '\0')
    {
        JSON_Value *js = json_parse_file(filename);
        JIParams *ret = jip_load_from_json(js);
        json_value_free(js);
        return ret;
    }

    FILE* jip_file = fopen(filename, "r");
    if (!jip_file)
    {
        fprintf(stderr, "Error opening JIP file\n");
        return NULL;
    }

    JIParams* jip = (JIParams*)xcalloc(1, sizeof(JIParams));
    if (fread(&jip->header, sizeof(JIPHeader), 1, jip_file) != 1)
    {
        fprintf(stderr, "Error reading header\n");
        goto read_jip_error;
    }

    jip->joint_params = (JIParam*)xmalloc(jip->header.n_joints * sizeof(JIParam));
    for (int i = 0; i < jip->header.n_joints; i++)
    {
        float params[3];
        if (fread(params, sizeof(float), 3, jip_file) != 3)
        {
            fprintf(stderr, "Error reading parameters\n");
            goto read_jip_error;
        }

        jip->joint_params[i].bandwidth = params[0];
        jip->joint_params[i].threshold = params[1];
        jip->joint_params[i].offset = params[2];
    }

    if (fclose(jip_file) != 0)
    {
        fprintf(stderr, "Error closing JIP file\n");
    }

    return jip;

read_jip_error:
    jip_free(jip);
    fclose(jip_file);
    return NULL;
}

void
jip_free(JIParams* jip)
{
    if (jip->joint_params)
    {
        xfree(jip->joint_params);
    }
    xfree(jip);
}
