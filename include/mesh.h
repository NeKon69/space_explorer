//
// Created by progamers on 7/23/25.
//

#ifndef SPACE_EXPLORER_MESH_H
#define SPACE_EXPLORER_MESH_H
#include <glad/glad.h>
#include <raw_memory.h>

#include "custom_deleters.h"
#include "rendering/vertex.h"
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
	void setup_object(const T& vertices, const Y& indices) {
		gen_opengl_data();
		indices_size = indices.size();
		glBindVertexArray(*vao);

		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(raw::vertex) * vertices.size(), vertices.data(),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(),
					 std::data(indices), GL_STATIC_DRAW);

		// positions and normals and tex coords
		GLsizei stride = sizeof(raw::vertex);
		GLsizei offset = offsetof(raw::vertex, position);
		// position attribute
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride, (void*)offset);
		glEnableVertexAttribArray(number_of_attr++);
		// texture coords attribute
		offset = offsetof(raw::vertex, tex_coord);
		glVertexAttribPointer(number_of_attr, 2, GL_FLOAT, GL_FALSE, stride, (void*)offset);
		glEnableVertexAttribArray(number_of_attr++);
		// normal attribute
		offset = offsetof(raw::vertex, normal);
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride, (void*)offset);
		glEnableVertexAttribArray(number_of_attr++);
		// tangent attribute
		offset = offsetof(raw::vertex, tangent);
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride, (void*)offset);
		glEnableVertexAttribArray(number_of_attr++);
		offset = offsetof(raw::vertex, bitangent);
		glVertexAttribPointer(number_of_attr, 3, GL_FLOAT, GL_FALSE, stride, (void*)offset);
		glEnableVertexAttribArray(number_of_attr++);
		glBindVertexArray(0);
	}

public:
	template<typename T, typename Y, typename U>
	mesh(const T& vertices, const Y& indices, const U& tex_coords,
		 bool normals_from_position = false)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)) {
		setup_object(vertices, indices, tex_coords, normals_from_position);
	}
	mesh(UI vertices_size, UI indices_size);

	void			 bind() const;
	void			 unbind() const;
	[[nodiscard]] UI get_index_count() const;
	[[nodiscard]] UI get_vao() const;
	[[nodiscard]] UI get_vbo() const;
	[[nodiscard]] UI get_ebo() const;
	[[nodiscard]] UI attr_num() const;
};
} // namespace raw

#endif // SPACE_EXPLORER_MESH_H
