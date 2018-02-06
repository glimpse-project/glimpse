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

#include "glimpse_config.h"
#include "parson.h"

void
gm_config_load(struct gm_logger *log, const char *json_buf,
               struct gm_ui_properties *props)
{
    JSON_Value *json = json_parse_string(json_buf);
    JSON_Object *config = json_object(json);

    for (size_t i = 0; i < json_object_get_count(config); ++i) {
        const char *name = json_object_get_name(config, i);

        for (int p = 0; p < props->n_properties; ++p) {
            struct gm_ui_property &prop = props->properties[p];

            // TODO: Decide whether we want to do bounds checking here.
            if (strcmp(name, prop.name) == 0) {
                JSON_Value *value =
                    json_object_get_value_at(config, i);

                switch (prop.type) {
                case GM_PROPERTY_INT: {
                    *prop.int_state.ptr = (int)json_value_get_number(value);
                    break;
                }

                case GM_PROPERTY_ENUM: {
                    const char *enum_val = json_value_get_string(value);
                    for (int j = 0; j < prop.enum_state.n_enumerants; ++j) {
                        const struct gm_ui_enumerant &enumerant =
                            prop.enum_state.enumerants[j];
                        if (strcmp(enum_val, enumerant.name) == 0) {
                            *prop.enum_state.ptr = enumerant.val;
                            break;
                        }
                    }
                    break;
                }

                case GM_PROPERTY_BOOL: {
                    *prop.bool_state.ptr = (bool)
                        json_value_get_boolean(value);
                    break;
                }

                case GM_PROPERTY_FLOAT: {
                    *prop.float_state.ptr = (float)
                        json_value_get_number(value);
                    break;
                }

                case GM_PROPERTY_FLOAT_VEC3: {
                    JSON_Array *array = json_value_get_array(value);
                    gm_assert(log, json_array_get_count(array) == 3,
                              "Invalid array size for vec3 in config");
                    for (int j = 0; j < 3; ++j) {
                        prop.vec3_state.ptr[j] = (float)
                            json_array_get_number(array, j);
                    }
                    break;
                }
                }
                break;
            }
        }
    }

    json_value_free(json);
}

char *
gm_config_save(struct gm_logger *log, struct gm_ui_properties *props)
{
    JSON_Value *json = json_value_init_object();
    JSON_Object *config = json_object(json);

    for (int p = 0; p < props->n_properties; ++p) {
        struct gm_ui_property &prop = props->properties[p];

        switch (prop.type) {
        case GM_PROPERTY_INT: {
            json_object_set_number(config, prop.name,
                                   (double)*prop.int_state.ptr);
            break;
        }

        case GM_PROPERTY_ENUM: {
            for (int i = 0; i < prop.enum_state.n_enumerants; ++i) {
                const struct gm_ui_enumerant &enumerant =
                    prop.enum_state.enumerants[i];
                if (enumerant.val == *prop.enum_state.ptr) {
                    json_object_set_string(config, prop.name, enumerant.name);
                    break;
                }
            }
            break;
        }

        case GM_PROPERTY_BOOL: {
            json_object_set_boolean(config, prop.name, *prop.bool_state.ptr);
            break;
        }

        case GM_PROPERTY_FLOAT: {
            json_object_set_number(config, prop.name,
                                   (double)*prop.float_state.ptr);
            break;
        }

        case GM_PROPERTY_FLOAT_VEC3: {
            JSON_Value *array_val = json_value_init_array();
            JSON_Array *array = json_array(array_val);
            for (int i = 0; i < 3; ++i) {
                json_array_append_number(array, (double)prop.vec3_state.ptr[i]);
            }
            json_object_set_value(config, prop.name, array_val);
            break;
        }
        }
    }

    char *retval = json_serialize_to_string_pretty(json);
    json_value_free(json);
    return retval;
}
