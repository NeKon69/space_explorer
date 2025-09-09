//
// Created by progamers on 9/9/25.
//

#pragma once
#include <glad/glad.h>

#include "deleters/custom_deleters.h"
#include "graphics/instanced_data.h"
#include "raw/unique_ptr.h"
namespace raw::graphics {
class instanced_data_buffer {
private:
	unique_ptr<uint32_t, deleters::gl_buffer> vbo;
	size_t									  capacity = 0;
	int										  stride   = 0;

public:
	instanced_data_buffer(uint32_t vao, uint32_t number_of_attr, size_t max_objects)
		: vbo(new UI(0)), capacity(max_objects) {
		glBindVertexArray(vao);
		// this sets up first instanced data (models), textures should be set up somewhere different
		glGenBuffers(1, vbo.get());
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		std::vector<graphics::instanced_data> vec(max_objects, {1.0f, 0, 0});
		;
		stride						= sizeof(instanced_data);
		constexpr int single_stride = sizeof(glm::vec4);
		glBufferData(GL_ARRAY_BUFFER, stride * max_objects, vec.data(), GL_DYNAMIC_DRAW);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, stride, nullptr);
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, stride,
							  (void*)(single_stride));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, stride,
							  (void*)(single_stride * 2));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, stride,
							  (void*)(single_stride * 3));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glBindVertexArray(0);
	}

	[[nodiscard]] uint32_t get_vbo() const {
		return *vbo;
	}
	[[nodiscard]] size_t get_size() const {
		return capacity * stride;
	}
};
} // namespace raw::graphics

