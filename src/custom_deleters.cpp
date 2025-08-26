//
// Created by progamers on 7/23/25.
//
#include "deleters/custom_deleters.h"

#include <glad/glad.h>

namespace raw::deleters {
    gl_array::gl_array(const raw::UI *data) {
        glDeleteVertexArrays(1, data);
        delete data;
    }

    gl_buffer::gl_buffer(const raw::UI *data) {
        glDeleteBuffers(1, data);
        delete data;
    }

    gl_texture::gl_texture(const raw::UI *data) {
        glDeleteTextures(1, data);
        delete data;
    }
} // namespace raw::deleter