//
// Created by progamers on 7/1/25.
//
#include "object.h"

#include <glad/glad.h>

#include "helper_macros.h"

namespace raw {

object::object(raw::object&& other) noexcept
	: vao(std::move(other.vao)),
	  vbo(std::move(other.vbo)),
	  ebo(std::move(other.ebo)),
	  indices_size(std::move(other.indices_size)) {}

object& object::operator=(raw::object&& other) noexcept {
	if (this != &other) {
		vao			 = std::move(other.vao);
		vbo			 = std::move(other.vbo);
		ebo			 = std::move(other.ebo);
		indices_size = std::move(other.indices_size);
	}
	return *this;
}

void object::__draw() const {
	glBindVertexArray(*vao);
	glDrawElements(GL_TRIANGLES, *indices_size, GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}

} // namespace raw