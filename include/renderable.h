//
// Created by progamers on 7/22/25.
//

#ifndef SPACE_EXPLORER_RENDERABLE_H
#define SPACE_EXPLORER_RENDERABLE_H
#include <glad/glad.h>
#include <raw_memory.h>

#include <glm/vec3.hpp>

#include "cuda_types/buffer.h"
#include "cuda_types/cuda_from_gl_data.h"
#include "debug.h"
#include "helper_macros.h"
#include "shader.h"

namespace raw {
struct gl_data_deleter_buffer {
	explicit gl_data_deleter_buffer(UI const* data) {
		glDeleteBuffers(1, data);
		delete data;
	}
};

struct gl_data_deleter_array {
	explicit gl_data_deleter_array(UI const* data) {
		glDeleteBuffers(1, data);
		delete data;
	}
};

enum class data_type { MODEL };
namespace predef {

PASSIVE_VALUE AMOUNT_OF_INSTANCED_DATA = 1U;
// I think this is the only way to do that, since opengl could not be initialized when we create
// that `data_types_creators`, so i need to put it in somewhere where it will be initialized at some
// controllable point
struct instanced_data {
	const std::unordered_map<UI, std::pair<data_type, int>> data_types = {
		{0, std::pair(data_type::MODEL, sizeof(glm::mat4))}};
	const std::unordered_map<UI, decltype(glGenVertexArrays)> data_types_creators = {
		{0, glGenVertexArrays}};
};
} // namespace predef

class renderable {
private:
	// Can't make it passive here for the same reason (which will cause copying of this thing, but
	// can't do anything with that)
	predef::instanced_data						inst_data;
	raw::unique_ptr<UI, gl_data_deleter_array>	vao;
	raw::unique_ptr<UI, gl_data_deleter_buffer> vbo;
	raw::unique_ptr<UI, gl_data_deleter_buffer> ebo;
	// I plan to create more of that kind of shit for:
	// 1) wooooow instanced rendering soo cool and funny
	// 2) Amount of time and memory that would be consumed otherwise would leave me with no choice
	std::array<raw::unique_ptr<UI, gl_data_deleter_buffer>, predef::AMOUNT_OF_INSTANCED_DATA>
							instanced_vbo;
	bool					created			   = false;
	int						num_of_obj_to_draw = 0;
	int						indices_size	   = 0;
	shared_ptr<raw::shader> shader;
#ifdef SPACE_EXPLORER_DEBUG
	void check_data_stability(size_t data_size) const;
#endif
	void gen_opengl_data(const std::initializer_list<UI>& sizes_of_buffers) const;

	template<typename T, typename Y>
	void setup_object(const T& vertices, const Y& indices,
					  const std::initializer_list<UI>& sizes_of_buffers,
					  bool							   normals_from_position = false) {
		gen_opengl_data(sizes_of_buffers);
		UI number_of_attr = 0;
		indices_size	  = indices.size();
		glBindVertexArray(*vao);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), std::data(vertices),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(),
					 std::data(indices), GL_STATIC_DRAW);

		// positions and normals
		constexpr GLsizei stride = 6 * sizeof(float);
		// position attribute
		glVertexAttribPointer(number_of_attr++, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
		glEnableVertexAttribArray(number_of_attr);
		// normal attribute
		glVertexAttribPointer(number_of_attr++, 3, GL_FLOAT, GL_FALSE, stride,
							  normals_from_position ? nullptr : (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(number_of_attr);

		// Start from 2, since 0 and 1 was already created above
		for (size_t i = 2; i < sizes_of_buffers.size(); ++i) {
			auto [data_type, obj_size] = inst_data.data_types.at(0);

			// I have genius setup here
			glBindBuffer(GL_ARRAY_BUFFER, *instanced_vbo[i]);
			switch (data_type) {
				using enum data_type;
			case MODEL: {
				auto ptr		   = sizes_of_buffers.begin();
				num_of_obj_to_draw = static_cast<int>(*ptr + number_of_attr);
				glVertexAttribPointer(number_of_attr++, 4, GL_FLOAT, GL_FALSE, obj_size, nullptr);
				glEnableVertexAttribArray(number_of_attr);
				glVertexAttribDivisor(number_of_attr, 1);

				glVertexAttribPointer(number_of_attr++, 4, GL_FLOAT, GL_FALSE, obj_size,
									  (void*)(obj_size / 4));
				glEnableVertexAttribArray(number_of_attr);
				glVertexAttribDivisor(number_of_attr, 1);

				glVertexAttribPointer(number_of_attr++, 4, GL_FLOAT, GL_FALSE, obj_size,
									  (void*)(obj_size / 2));
				glEnableVertexAttribArray(number_of_attr);
				glVertexAttribDivisor(number_of_attr, 1);

				glVertexAttribPointer(number_of_attr++, 4, GL_FLOAT, GL_FALSE, obj_size,
									  (void*)(obj_size / 4 * 3));
				glEnableVertexAttribArray(number_of_attr);
				glVertexAttribDivisor(number_of_attr, 1);

				break;
			}
			default:
				// For now nothing here
				break;
			}
		}
		glBindVertexArray(0);
	}
	void setup_object(const std::initializer_list<UI>& sizes_of_buffers,
					  bool							   normals_from_positions = false);

public:
	renderable() = default;

	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	/**
	 * @param vertices vertices data for gpu buffer
	 * @param indices indices data for gpu buffer
	 * @param sizes_of_buffers list of sizes for all buffers (NOTE: THAT INCLUDES EVEN VERTICES AND
	 * INDICES SIZES)
	 * @param normals_from_position mark this true if you think that normals are the same as the
	 * positions
	 */
	renderable(const T& vertices, const Y& indices,
			   // Into this you should pass also sizes of vertices and indices
			   const std::initializer_list<UI>& sizes_of_buffers,
			   bool								normals_from_position = false)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)) {
#ifdef SPACE_EXPLORER_DEBUG
		check_data_stability(sizes_of_buffers.size());
#endif
		setup_object(vertices, indices, sizes_of_buffers, normals_from_position);
	}
	renderable(const std::initializer_list<UI>& sizes_of_buffers);
	void set_shader(const shared_ptr<raw::shader>& sh);
	void draw() const;
};
} // namespace raw
#endif // SPACE_EXPLORER_RENDERABLE_H
