//
// Created by progamers on 7/22/25.
//
#include "renderable.h"
namespace raw {

void renderable::check_data_stability(size_t data_size) const {
	if (data_size != inst_data.data_types_creators.size() ||
		inst_data.data_types_creators.size() - 2 != predef::AMOUNT_OF_INSTANCED_DATA ||
		predef::AMOUNT_OF_INSTANCED_DATA != inst_data.data_types.size() - 2) {
		throw std::runtime_error(
			"[Error] Amount of instanced data is not equal! Check renderable class or what you are passing to it.");
	}
}

void renderable::gen_opengl_data(const std::initializer_list<UI>& sizes_of_buffers) const {
	glGenVertexArrays(1, vao.get());
	// first (vertices)
	glGenBuffers(1, vbo.get());
	// second (indices)
	glGenBuffers(1, ebo.get());
	// then this should be 3
	for (size_t i = 2; i < sizes_of_buffers.size(); ++i) {
		// that looks crazy, but, trust me, that just makes call for correct function
		inst_data.data_types_creators.at(static_cast<unsigned int>(i - 2))(
			1, instanced_vbo[i - 2].get());
	}
}

void renderable::setup_object(const std::initializer_list<UI>& sizes_of_buffers,
							  bool							   normals_from_position) {
	auto				   begin = sizes_of_buffers.begin();
	std::vector<UI>		   indices_dummy(*begin++);
	std::vector<glm::vec3> vertices_dummy(*begin);
	setup_object(vertices_dummy, indices_dummy, sizes_of_buffers, normals_from_position);
}

renderable::renderable(const std::initializer_list<UI>& sizes_of_buffers)
	: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)) {
#ifdef SPACE_EXPLORER_DEBUG
	check_data_stability(sizes_of_buffers.size());
#endif
	setup_object(sizes_of_buffers);
}
void renderable::set_shader(const shared_ptr<raw::shader>& sh) {
	shader = sh;
}
void renderable::draw() const {
	shader->use();
	glBindVertexArray(*vao);
	glDrawElementsInstanced(GL_TRIANGLES, indices_size, GL_UNSIGNED_INT, nullptr,
							num_of_obj_to_draw);
}
} // namespace raw