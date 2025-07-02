//
// Created by progamers on 7/1/25.
//
#include "object.h"

#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

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

void object::rotate(const float degree, const glm::vec3& rotation) {
	transformation = glm::rotate(transformation, degree, rotation);
}
void object::move(const glm::vec3& destination) {
	transformation = glm::translate(transformation, destination);
}
void object::scale(const glm::vec3& factor) {
	transformation = glm::scale(transformation, factor);
}
void object::draw(bool should_reset) {
	shader->use();
	shader->set_mat4("model", glm::value_ptr(transformation));
	__draw();
    if(should_reset) {
        reset();
    }
}

void object::reset() {
	transformation = glm::mat4(1.0f);
}

} // namespace raw