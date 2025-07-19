//
// Created by progamers on 7/18/25.
//

#include "objects/sphere.h"

#include <glm/gtc/type_ptr.hpp>

namespace raw {
sphere::sphere(const raw::shared_ptr<raw::shader>& shader, float radius)
	: indices(predef::MAXIMUM_AMOUNT_OF_INDICES),
	  vertices(predef::MAXIMUM_AMOUNT_OF_VERTICES),
	  obj(vertices, predef::MAXIMUM_AMOUNT_OF_VERTICES, indices, predef::MAXIMUM_AMOUNT_OF_INDICES),
	  gen(obj.get_vbo(), obj.get_ebo(), predef::BASIC_STEPS, radius) {
	obj.set_shader(shader);
}

void sphere::rotate(const float degree, const glm::vec3& axis) {
	obj.rotate(degree, axis);
}
void sphere::move(const glm::vec3& destination) {
	obj.move(destination);
}
void sphere::scale(const glm::vec3& factor) {
	obj.scale(factor);
}
void sphere::draw(decltype(drawing_method::drawing_method) drawing_method, bool should_reset) {
	obj.draw(drawing_method, should_reset);
}
void sphere::set_shader(const raw::shared_ptr<raw::shader>& sh) {
	obj.set_shader(sh);
}

void sphere::reset() {
	obj.reset();
}
} // namespace raw