//
// Created by progamers on 7/23/25.
//
#include "mesh.h"

namespace raw {
void mesh::gen_opengl_data() const {
	glGenVertexArrays(1, vao.get());
	glGenBuffers(1, vbo.get());
	glGenBuffers(1, texture_coords_vbo.get());
	glGenBuffers(1, ebo.get());
}
mesh::mesh(const raw::UI vertices_size, const raw::UI indices_size, bool normals_from_position)
	: vao(new UI(0)), vbo(new UI(0)), texture_coords_vbo(new UI(0)), ebo(new UI(0)) {
	std::vector<glm::vec3> vertices(vertices_size);
	std::vector<UI>		   indices(indices_size);
	// Amount of texture coords is the same as amount of vertices
	std::vector<glm::vec2> tex_coords(vertices_size);
	setup_object(vertices, indices, tex_coords, normals_from_position);
}
void mesh::bind() const {
	glBindVertexArray(*vao);
}
void mesh::unbind() const {
	glBindVertexArray(0);
}

UI mesh::get_index_count() const {
	return indices_size;
}
UI mesh::get_vao() const {
	return *vao;
}

UI mesh::get_vbo() const {
	return *vbo;
}
UI mesh::get_tex_coords_vbo() const {
	return *texture_coords_vbo;
}
UI mesh::get_ebo() const {
	return *ebo;
}
UI mesh::attr_num() const {
	return number_of_attr;
}

} // namespace raw