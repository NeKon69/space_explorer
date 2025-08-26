//
// Created by progamers on 7/1/25.
//
#include "../include/z_unused/object.h"

#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

#include "../include/cuda_types/error.h"

namespace raw {
    namespace drawing_method {
        void basic(UI *vao, UI indices_size) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
        }

        void lines(UI *vao, UI indices_size) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        void points(UI *vao, UI indices_size) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        void transparent_alpha(const UI *vao, UI indices_size) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_FALSE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
        }

        void always_visible(UI *vao, UI indices_size) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDepthFunc(GL_LESS);
        }

        void backface_cull(UI *vao, UI indices_size) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_CULL_FACE);
        }

        void polygon_offset_fill(UI *vao, UI indices_size) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_POLYGON_OFFSET_FILL);
        }

        void stencil_mask_equal_1(UI *vao, UI indices_size) {
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_STENCIL_TEST);
        }

        void depth_write_disabled(UI *vao, UI indices_size) {
            glDepthMask(GL_FALSE);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDepthMask(GL_TRUE);
        }

        void color_mask_red_only(UI *vao, UI indices_size) {
            glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        }

        void blend_additive(UI *vao, UI indices_size) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glEnable(GL_DEPTH_TEST);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_CULL_FACE);
            glDepthFunc(GL_LESS);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_FALSE);

            glBindVertexArray(*vao);
            glDrawElements(GL_TRIANGLES, static_cast<int>(indices_size), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
        }
    } // namespace drawing_method

    void object::gen_opengl_data() const {
        glGenVertexArrays(1, vao.get());
        glGenBuffers(1, vbo.get());
        glGenBuffers(1, ebo.get());
    }

    object::object(raw::object &&other) noexcept
        : vao(std::move(other.vao)),
          vbo(std::move(other.vbo)),
          ebo(std::move(other.ebo)),
          indices_size(other.indices_size) {
    }

    object &object::operator=(raw::object &&other) noexcept {
        if (this != &other) {
            vao = std::move(other.vao);
            vbo = std::move(other.vbo);
            ebo = std::move(other.ebo);
            indices_size = other.indices_size;
        }
        return *this;
    }

    void object::__draw() const {
        glBindVertexArray(*vao);
        glDrawElements(GL_TRIANGLES, indices_size, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    void object::rotate(const float degree, const glm::vec3 &axis) {
        transformation = glm::rotate(transformation, degree, axis);
    }

    void object::move(const glm::vec3 &destination) {
        transformation = glm::translate(transformation, destination);
    }

    void object::scale(const glm::vec3 &factor) {
        transformation = glm::scale(transformation, factor);
    }

    void object::rotate_around(const float degree, const glm::vec3 &axis,
                               const glm::vec3 &distance_to_object) {
        move(distance_to_object);
        rotate(degree, axis);
    }

    void object::draw(decltype(drawing_method::drawing_method) drawing_method, bool should_reset) {
        shader->use();
        shader->set_mat4("model", glm::value_ptr(transformation));
        drawing_method(vao.get(), indices_size);
        if (should_reset) {
            reset();
        }
    }

    void object::set_shader(const raw::shared_ptr<raw::shader> &sh) {
        shader = sh;
        shader->use();
    }

    void object::reset() {
        transformation = glm::mat4(1.0f);
    }
} // namespace raw