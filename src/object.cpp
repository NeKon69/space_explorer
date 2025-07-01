//
// Created by progamers on 7/1/25.
//
#include "object.h"

#include <glad/glad.h>

#include "helper_macros.h"
namespace raw {
object::object(const float* vertices, const unsigned int* indices)
	: vao(0), vbo(0), ebo(0), deleted_buffers(nullptr) {
    deleted_buffers = new bool(true);
	setup_object(vertices, indices);
}
void object::setup_object(const float* vertices, const unsigned int* indices) {
	glGenVertexArrays(1, vao);
	glGenBuffers(1, vbo);
	glGenBuffers(1, ebo);

	glBindVertexArray(*vao);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 9 * 36, vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 36, indices, GL_STATIC_DRAW);

	constexpr GLsizei stride = 9 * sizeof(float);
	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glEnableVertexAttribArray(0);
	// normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

object::object(const raw::object& other)
	: vao(other.vao), vbo(other.vbo), ebo(other.ebo), deleted_buffers(other.deleted_buffers) {}
object::object(raw::object&& other) noexcept
	: vao(other.vao), vbo(other.vbo), ebo(other.ebo), deleted_buffers(other.deleted_buffers) {
	other.vao			  = 0;
	other.vbo			  = 0;
	other.ebo			  = 0;
	other.deleted_buffers = true;
}

object& object::operator=(const raw::object& other) {
	if (this != &other) {
		vao = other.vao;
		vbo = other.vbo;
		ebo = other.ebo;
	}
	return *this;
}

object& object::operator=(raw::object&& other) noexcept {
	if (this != &other) {
		vao		  = std::move(other.vao);
		vbo		  = std::move(other.vbo);
		ebo		  = std::move(other.ebo);
		other.vao = 0;
		other.vbo = 0;
		other.ebo = 0;
	}
	return *this;
}

object::~object() noexcept {
	if (vao) {
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}
	if (vbo) {
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}
	if (ebo) {
		glDeleteBuffers(1, &ebo);
		ebo = 0;
	}
}

} // namespace raw