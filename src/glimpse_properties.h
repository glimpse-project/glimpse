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

#include <pthread.h>
#include <stdbool.h>
#include <string.h>

#include "parson.h"

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
    GM_PROPERTY_STRING,
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
        struct {
            /* XXX: note that there's currently no locking so there's
             * a risk we'll get into trouble racing between
             * getting/setting at some point...
             */
            char **ptr;
            const char *(*get)(struct gm_ui_property *prop);
            void (*set)(struct gm_ui_property *prop, const char *string);
        } string_state;
    };
    bool read_only;
};

struct gm_ui_properties {
    pthread_mutex_t lock;
    int n_properties;
    struct gm_ui_property *properties;
};


#ifdef __cplusplus
extern "C" {
#endif

#define GM_DECLARE_SCALAR_PROP_GETTER(NAME, CTYPE) \
static inline CTYPE \
gm_prop_get_##NAME(struct gm_ui_property *prop) \
{ \
    if (prop->NAME##_state.get) \
        return prop->NAME##_state.get(prop); \
    else \
        return *prop->NAME##_state.ptr; \
}
GM_DECLARE_SCALAR_PROP_GETTER(int, int)
GM_DECLARE_SCALAR_PROP_GETTER(bool, bool)
GM_DECLARE_SCALAR_PROP_GETTER(enum, int)
GM_DECLARE_SCALAR_PROP_GETTER(float, float)

#define GM_DECLARE_SCALAR_PROP_SETTER(NAME, CTYPE) \
static inline void \
gm_prop_set_##NAME(struct gm_ui_property *prop, CTYPE val) \
{ \
    if (prop->NAME##_state.set) \
        prop->NAME##_state.set(prop, val); \
    else \
        *(prop->NAME##_state.ptr) = val; \
}
GM_DECLARE_SCALAR_PROP_SETTER(int, int)
GM_DECLARE_SCALAR_PROP_SETTER(bool, bool)
GM_DECLARE_SCALAR_PROP_SETTER(enum, int)
GM_DECLARE_SCALAR_PROP_SETTER(float, float)

static inline void
gm_prop_get_vec3(struct gm_ui_property *prop, float *out)
{
    if (prop->vec3_state.get)
        prop->vec3_state.get(prop, out);
    else
        memcpy(out, prop->vec3_state.ptr, sizeof(float) * 3);
}

static inline void
gm_prop_set_vec3(struct gm_ui_property *prop, float *vec3)
{
    if (prop->vec3_state.set)
        prop->vec3_state.set(prop, vec3);
    else
        memcpy(prop->vec3_state.ptr, vec3, sizeof(float) * 3);
}

static inline const char *
gm_prop_get_string(struct gm_ui_property *prop)
{
    if (prop->string_state.get)
        return prop->string_state.get(prop);
    else
        return *prop->string_state.ptr;
}

static inline void
gm_prop_set_string(struct gm_ui_property *prop, const char *string)
{
    if (prop->string_state.set)
        prop->string_state.set(prop, string);
    else {
        free(*prop->string_state.ptr);
        *prop->string_state.ptr = NULL;
        if (string)
            *prop->string_state.ptr = strdup(string);
    }
}

static inline void
gm_prop_set_switch(struct gm_ui_property *prop)
{
    if (prop->switch_state.set)
        prop->switch_state.set(prop);
    else
        *(prop->switch_state.ptr) = true;
}

static inline void
gm_prop_set_enum_by_name(struct gm_ui_property *prop, const char *name)
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

static inline const char *
gm_prop_set_enum_name(struct gm_ui_property *prop, int val)
{
    for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
        const struct gm_ui_enumerant *enumerant =
            &prop->enum_state.enumerants[i];
        if (enumerant->val == val)
            return enumerant->name;
    }
    return NULL;
}

static inline int
gm_props_lookup_id(struct gm_ui_properties *props, const char *name)
{
    for (int i = 0; i < props->n_properties; i++) {
        if (strcmp(props->properties[i].name, name) == 0)
            return i;
    }
    return -1;
}

static inline struct gm_ui_property *
gm_props_lookup(struct gm_ui_properties *props, const char *name)
{
    int id = gm_props_lookup_id(props, name);
    if (id >= 0)
        return &props->properties[id];
    else
        return NULL;
}

#define GM_DECLARE_SCALAR_PROPS_GETTER(NAME, CTYPE) \
static inline CTYPE \
gm_props_get_##NAME(struct gm_ui_properties *props, \
                    const char *name) \
{ \
    struct gm_ui_property *prop = gm_props_lookup(props, name); \
    if (prop) \
        return gm_prop_get_##NAME(prop); \
    else \
        return 0; \
}
GM_DECLARE_SCALAR_PROPS_GETTER(int, int)
GM_DECLARE_SCALAR_PROPS_GETTER(bool, bool)
GM_DECLARE_SCALAR_PROPS_GETTER(enum, int)
GM_DECLARE_SCALAR_PROPS_GETTER(float, float)

#define GM_DECLARE_SCALAR_PROPS_SETTER(NAME, CTYPE) \
static inline void \
gm_props_set_##NAME(struct gm_ui_properties *props, \
                    const char *name, \
                    CTYPE val) \
{ \
    struct gm_ui_property *prop = gm_props_lookup(props, name); \
    if (prop) \
        gm_prop_set_##NAME(prop, val); \
}
GM_DECLARE_SCALAR_PROPS_SETTER(int, int)
GM_DECLARE_SCALAR_PROPS_SETTER(bool, bool)
GM_DECLARE_SCALAR_PROPS_SETTER(enum, int)
GM_DECLARE_SCALAR_PROPS_SETTER(float, float)

static inline void
gm_props_get_vec3(struct gm_ui_properties *props,
                  const char *name,
                  float *out)
{
    struct gm_ui_property *prop = gm_props_lookup(props, name);
    if (prop)
        gm_prop_get_vec3(prop, out);
    else
        out[0] = out[1] = out[2] = 0;
}

static inline void
gm_props_set_vec3(struct gm_ui_properties *props,
                  const char *name,
                  float *vec3)
{
    struct gm_ui_property *prop = gm_props_lookup(props, name);
    if (prop)
        gm_prop_set_vec3(prop, vec3);
}

static inline const char *
gm_props_get_string(struct gm_ui_properties *props,
                    const char *name)
{
    struct gm_ui_property *prop = gm_props_lookup(props, name);
    if (prop)
        return gm_prop_get_string(prop);
    else
        return NULL;
}

static inline void
gm_props_set_string(struct gm_ui_properties *props,
                    const char *name,
                    const char *string)
{
    struct gm_ui_property *prop = gm_props_lookup(props, name);
    if (prop)
        gm_prop_set_string(prop, string);
}

static inline void
gm_props_set_switch(struct gm_ui_properties *props,
                    const char *name)
{
    struct gm_ui_property *prop = gm_props_lookup(props, name);
    if (prop)
        gm_prop_set_switch(prop);
}

void
gm_props_from_json(struct gm_logger *log,
                   struct gm_ui_properties *props,
                   JSON_Value *props_object);

void
gm_props_to_json(struct gm_logger *log,
                 struct gm_ui_properties *props,
                 JSON_Value *props_object);

#ifdef __cplusplus
}
#endif
