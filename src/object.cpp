//
// Created by progamers on 7/1/25.
//
#include "object.h"

#include <glad/glad.h>

#include "helper_macros.h"

namespace raw {

object::object(raw::object&& other) noexcept
	: vao(std::move(other.vao)), vbo(std::move(other.vbo)), ebo(std::move(other.ebo)),
      indices_size(std::move(other.indices_size)) {}

object& object::operator=(const raw::object& other) {
	if (this != &other) {
		vao = other.vao;
		vbo = other.vbo;
		ebo = other.ebo;
        indices_size = other.indices_size;
	}
	return *this;
}

object& object::operator=(raw::object&& other) noexcept {
	if (this != &other) {
		vao		   = std::move(other.vao);
		vbo		   = std::move(other.vbo);
		ebo		   = std::move(other.ebo);
        indices_size = std::move(other.indices_size);
	}
	return *this;
}

object::~object() noexcept {
    // I didn't make the destructor template for my smart pointers, so I need to have that ugly code here
	if (vao) {
		glDeleteVertexArrays(1, vao.get());
	}
	if (vbo) {
		glDeleteBuffers(1, vbo.get());
	}
	if (ebo) {
		glDeleteBuffers(1, ebo.get());
	}
}

void object::__draw() const {
	glBindVertexArray(*vao);
	glDrawElements(GL_TRIANGLES, *indices_size, GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}

} // namespace raw