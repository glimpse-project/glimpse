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

#pragma once

#include <stdbool.h>
#include <string.h>
#include "glimpse_log.h"


enum gm_rotation {
  GM_ROTATION_0 = 0,
  GM_ROTATION_90 = 1,
  GM_ROTATION_180 = 2,
  GM_ROTATION_270 = 3
};

enum gm_property_type {
    GM_PROPERTY_INT,
    GM_PROPERTY_ENUM,
    GM_PROPERTY_BOOL,
    GM_PROPERTY_SWITCH,
    GM_PROPERTY_FLOAT,
    GM_PROPERTY_FLOAT_VEC3,
};

struct gm_ui_enumerant {
    const char *name;
    const char *desc;
    int val;
};

struct gm_ui_property {
    void *object; // thing the property is for
    const char *name;
    const char *desc;
    enum gm_property_type type;
    union {
        struct {
            int *ptr;
            int min;
            int max;
            int (*get)(struct gm_ui_property *prop);
            void (*set)(struct gm_ui_property *prop, int val);
        } int_state;
        struct {
            int *ptr;
            int n_enumerants;
            const struct gm_ui_enumerant *enumerants;
            int (*get)(struct gm_ui_property *prop);
            void (*set)(struct gm_ui_property *prop, int val);
        } enum_state;
        struct {
            bool *ptr;
            bool (*get)(struct gm_ui_property *prop);
            void (*set)(struct gm_ui_property *prop, bool val);
        } bool_state;
        struct {
            bool *ptr;
            void (*set)(struct gm_ui_property *prop);
        } switch_state;
        struct {
            float *ptr;
            float min;
            float max;
            float (*get)(struct gm_ui_property *prop);
            void (*set)(struct gm_ui_property *prop, float val);
        } float_state;
        struct {
            float *ptr;
            const char *components[3];
            float min[3];
            float max[3];
            void (*get)(struct gm_ui_property *prop, float *out);
            void (*set)(struct gm_ui_property *prop, float *val);
        } vec3_state;
    };
    bool read_only;
};

#define GM_DECLARE_SCALAR_GETTER(NAME, CTYPE) \
inline CTYPE gm_prop_get_##NAME(struct gm_ui_property *prop) \
{ \
    if (prop->NAME##_state.get) \
        return prop->NAME##_state.get(prop); \
    else \
        return *prop->NAME##_state.ptr; \
}
GM_DECLARE_SCALAR_GETTER(int, int)
GM_DECLARE_SCALAR_GETTER(bool, bool)
GM_DECLARE_SCALAR_GETTER(enum, int)
GM_DECLARE_SCALAR_GETTER(float, float)

#define GM_DECLARE_SCALAR_SETTER(NAME, CTYPE) \
inline void gm_prop_set_##NAME(struct gm_ui_property *prop, CTYPE val) \
{ \
    if (prop->NAME##_state.set) \
        prop->NAME##_state.set(prop, val); \
    else \
        *(prop->NAME##_state.ptr) = val; \
}
GM_DECLARE_SCALAR_SETTER(int, int)
GM_DECLARE_SCALAR_SETTER(bool, bool)
GM_DECLARE_SCALAR_SETTER(enum, int)
GM_DECLARE_SCALAR_SETTER(float, float)

inline void gm_prop_get_vec3(struct gm_ui_property *prop, float *out)
{
    if (prop->vec3_state.get)
        prop->vec3_state.get(prop, out);
    else
        memcpy(out, prop->vec3_state.ptr, sizeof(float) * 3);
}

inline void gm_prop_set_vec3(struct gm_ui_property *prop, float *vec3)
{
    if (prop->vec3_state.set)
        prop->vec3_state.set(prop, vec3);
    else
        memcpy(prop->vec3_state.ptr, vec3, sizeof(float) * 3);
}

inline void gm_prop_set_switch(struct gm_ui_property *prop)
{
    if (prop->switch_state.set)
        prop->switch_state.set(prop);
    else
        *(prop->switch_state.ptr) = true;
}

inline void gm_prop_set_enum_by_name(struct gm_ui_property *prop, const char *name)
{
    for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
        const struct gm_ui_enumerant *enumerant =
            &prop->enum_state.enumerants[i];
        if (strcmp(name, enumerant->name) == 0) {
            gm_prop_set_enum(prop, enumerant->val);
            break;
        }
    }
}

inline const char *gm_prop_set_enum_name(struct gm_ui_property *prop, int val)
{
    for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
        const struct gm_ui_enumerant *enumerant =
            &prop->enum_state.enumerants[i];
        if (enumerant->val == val)
            return enumerant->name;
    }
    return NULL;
}

/* During development and testing it's convenient to have direct tuneables
 * we can play with at runtime...
 */
struct gm_ui_properties {
    pthread_mutex_t lock;
    int n_properties;
    struct gm_ui_property *properties;
};

#ifdef __cplusplus
extern "C" {
#endif

void gm_config_load(struct gm_logger *log,
                    const char *json_buf,
                    struct gm_ui_properties *props);
char *gm_config_save(struct gm_logger *log,
                     struct gm_ui_properties *props);

#ifdef __cplusplus
}
#endif
