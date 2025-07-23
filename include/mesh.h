//
// Created by progamers on 7/23/25.
//

#ifndef SPACE_EXPLORER_MESH_H
#define SPACE_EXPLORER_MESH_H
#include <glad/glad.h>
#include <raw_memory.h>

#include "custom_deleters.h"
#include "shader.h"
namespace raw {

class mesh {
private:
	unique_ptr<UI, deleter::gl_array>  vao;
	unique_ptr<UI, deleter::gl_buffer> vbo;
	unique_ptr<UI, deleter::gl_buffer> ebo;
	UI								   indices_size	  = 0;
	UI								   number_of_attr = 0;

	void gen_opengl_data() const;
	template<typename T, typename Y>
	void setup_object(const T& vertices, const Y& indices, bool normals_from_position) {
		gen_opengl_data();
		indices_size = indices.size();
		glBindVertexArray(*vao);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), std::data(vertices),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(),
					 std::data(indices), GL_STATIC_DRAW);

		// positions and normals
		constexpr GLsizei stride = 6 * sizeof(float);
		// position attribute
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
		glEnableVertexAttribArray(number_of_attr++);
		// normal attribute
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride,
							  normals_from_position ? nullptr : (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(number_of_attr++);
	}

public:
	template<typename T, typename Y>
	mesh(const T& vertices, const Y& indices, bool normals_from_position = false)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)) {
		setup_object(vertices, indices, normals_from_position);
	}
	mesh(UI vertices_size, UI indices_size, bool normals_from_position = false);

	void			 bind() const;
	void			 unbind() const;
	[[nodiscard]] UI get_index_count() const;
	[[nodiscard]] UI get_vbo() const;
	[[nodiscard]] UI get_ebo() const;
	[[nodiscard]] UI attr_num() const;
};
} // namespace raw

#endif // SPACE_EXPLORER_MESH_H
