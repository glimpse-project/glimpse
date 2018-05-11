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

#include "glimpse_properties.h"
#include "parson.h"

void
gm_props_from_json(struct gm_logger *log,
                   struct gm_ui_properties *props,
                   JSON_Value *props_object)
{
    JSON_Object *config = json_object(props_object);

    for (size_t i = 0; i < json_object_get_count(config); ++i) {
        const char *name = json_object_get_name(config, i);

        for (int p = 0; p < props->n_properties; ++p) {
            struct gm_ui_property *prop = &props->properties[p];

            if (prop->read_only)
                continue;

            if (strcmp(name, prop->name) != 0)
                continue;

            JSON_Value *value =
                json_object_get_value_at(config, i);

            switch (prop->type) {
            case GM_PROPERTY_INT:
                gm_prop_set_int(prop, json_value_get_number(value));
                break;
            case GM_PROPERTY_ENUM:
                gm_prop_set_enum_by_name(prop, json_value_get_string(value));
                break;
            case GM_PROPERTY_BOOL:
                gm_prop_set_bool(prop, json_value_get_boolean(value));
                break;
            case GM_PROPERTY_FLOAT:
                gm_prop_set_float(prop, json_value_get_number(value));
                break;
            case GM_PROPERTY_FLOAT_VEC3: {
                JSON_Array *array = json_value_get_array(value);
                gm_assert(log, json_array_get_count(array) == 3,
                          "Invalid array size for vec3 in config");
                float vec3[3];
                for (int j = 0; j < 3; ++j)
                    vec3[j] = json_array_get_number(array, j);
                gm_prop_set_vec3(prop, vec3);
                break;
            }
            case GM_PROPERTY_SWITCH:
                // SKIP
                break;
            case GM_PROPERTY_STRING:
                gm_prop_set_string(prop, json_value_get_string(value));
                break;
            }
        }
    }
}

void
gm_props_to_json(struct gm_logger *log,
                 struct gm_ui_properties *props,
                 JSON_Value *props_object)
{
    JSON_Object *config = json_object(props_object);

    for (int p = 0; p < props->n_properties; ++p) {
        struct gm_ui_property *prop = &props->properties[p];

        if (prop->read_only)
            continue;

        switch (prop->type) {
        case GM_PROPERTY_INT:
            json_object_set_number(config, prop->name,
                                   gm_prop_get_int(prop));
            break;
        case GM_PROPERTY_ENUM: {
            int val = gm_prop_get_enum(prop);
            for (int i = 0; i < prop->enum_state.n_enumerants; i++) {
                const struct gm_ui_enumerant *enumerant =
                    &prop->enum_state.enumerants[i];
                if (enumerant->val == val) {
                    json_object_set_string(config, prop->name, enumerant->name);
                    break;
                }
            }
            break;
        }
        case GM_PROPERTY_BOOL:
            json_object_set_boolean(config, prop->name, gm_prop_get_bool(prop));
            break;
        case GM_PROPERTY_FLOAT:
            json_object_set_number(config, prop->name, gm_prop_get_float(prop));
            break;
        case GM_PROPERTY_FLOAT_VEC3: {
            JSON_Value *array_val = json_value_init_array();
            JSON_Array *array = json_array(array_val);
            float vec3[3];
            gm_prop_get_vec3(prop, vec3);
            for (int i = 0; i < 3; ++i) {
                json_array_append_number(array, vec3[i]);
            }
            json_object_set_value(config, prop->name, array_val);
            break;
        }
        case GM_PROPERTY_SWITCH:
            // SKIP
            break;
        case GM_PROPERTY_STRING:
            json_object_set_string(config, prop->name, gm_prop_get_string(prop));
            break;
        }
    }
}
