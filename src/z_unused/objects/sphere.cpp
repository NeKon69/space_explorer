//
// Created by progamers on 7/18/25.
//

#include "z_unused/objects/sphere.h"

#include <glm/gtc/type_ptr.hpp>

#include "rendering/shader/shader.h"

namespace raw::z_unused::objects {
	sphere::sphere(const std::shared_ptr<raw::rendering::shader::shader> &shader, float radius)
		: indices(sphere_generation::predef::MAXIMUM_AMOUNT_OF_INDICES),
		  vertices(sphere_generation::predef::MAXIMUM_AMOUNT_OF_VERTICES),
		  obj(vertices, sphere_generation::predef::MAXIMUM_AMOUNT_OF_VERTICES, indices,
		      sphere_generation::predef::MAXIMUM_AMOUNT_OF_INDICES),
		  gen(obj.get_vbo(), obj.get_ebo(), std::make_shared<cuda_types::cuda_stream>()) {
		obj.set_shader(shader);
		obj.scale(glm::vec3(radius));
		// Acted as a dummy to create obj, so they can be cleared
		indices.clear();
		indices.shrink_to_fit();
		vertices.clear();
		vertices.shrink_to_fit();
	}

	void sphere::rotate(const float degree, const glm::vec3 &axis) {
		obj.rotate(degree, axis);
	}

	void sphere::move(const glm::vec3 &destination) {
		obj.move(destination);
	}

	void sphere::scale(const glm::vec3 &factor) {
		obj.scale(factor);
	}

	void sphere::rotate_around(const float degree, const glm::vec3 &axis,
	                           const glm::vec3 &distance_to_object) {
		obj.rotate_around(degree, axis, distance_to_object);
	}

	void sphere::draw(decltype(drawing_method::drawing_method) drawing_method, bool should_reset) {
		obj.draw(drawing_method, should_reset);
	}

	void sphere::set_shader(const std::shared_ptr<raw::rendering::shader::shader> &sh) {
		obj.set_shader(sh);
	}

	void sphere::reset() {
		obj.reset();
	}
} // namespace raw::z_unused::objects
