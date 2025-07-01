//
// Created by progamers on 7/1/25.
//
#include "objects/cube.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace raw {

cube::cube(raw::shader& sh) : object(this->cube_pos, this->indices), shader(sh) {}

void cube::rotate(const float degree, const glm::vec3& rotation) {
	transformation = glm::rotate(transformation, degree, rotation);
}
void cube::move(const glm::vec3& destination) {
	transformation = glm::translate(transformation, destination);
}
void cube::scale(const glm::vec3& factor) {
	transformation = glm::scale(transformation, factor);
}
void cube::draw() {
	shader.use();
	shader.set_mat4("model", glm::value_ptr(transformation));
	__draw();
}

void cube::reset() {
    transformation = glm::mat4(1.0f);
}

} // namespace raw