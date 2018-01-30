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

#include <stdlib.h>

#include <epoxy/gl.h>

#include "xalloc.h"
#include "glimpse_log.h"

unsigned int
gm_gl_load_shader(struct gm_logger *log,
                  int type,
                  const char *source,
                  char **err)
{
    GLuint shader = glCreateShader(type);
    if (!shader)
        return 0;

    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled) {
        GLint info_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len) {
            char *buf = (char *)malloc(info_len);
            if (buf) {
                glGetShaderInfoLog(shader, info_len, NULL, buf);
                gm_throw(log, err, "Could not compile shader %d:\n%s\n",
                         type, buf);
                free(buf);
            }
            glDeleteShader(shader);
            shader = 0;
        }
    }

    return shader;
}

unsigned int
gm_gl_create_program(struct gm_logger *log,
                     const char *vertex_source,
                     const char *fragment_source,
                     char **err)
{
    GLuint vertex_shader = gm_gl_load_shader(log, GL_VERTEX_SHADER, vertex_source, err);
    if (!vertex_shader)
        return 0;

    GLuint fragment_shader = gm_gl_load_shader(log, GL_FRAGMENT_SHADER, fragment_source, err);
    if (!fragment_shader) {
        glDeleteShader(vertex_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    if (!program) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
        GLint buf_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &buf_length);
        if (buf_length) {
            char *buf = (char *)malloc(buf_length);
            if (buf) {
                glGetProgramInfoLog(program, buf_length, NULL, buf);
                gm_throw(log, err, "Could not link program:\n%s\n", buf);
                free(buf);
            } else {
                gm_throw(log, err, "Failed to link program: Could not get error log");
            }
        }
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}


