//
// Created by progamers on 7/7/25.
//

#include "sphere_generation/icosahedron_data_manager.h"

#include <numbers>

#include "core/clock.h"
#include "cuda_types/buffer.h"
#include "sphere_generation/generation_context.h"
#include "sphere_generation/kernel_launcher.h"

namespace raw::sphere_generation {
inline constexpr float GOLDEN_RATIO = std::numbers::phi_v<float>;
inline constexpr float PI			= std::numbers::pi_v<float>;

icosahedron_data_manager::icosahedron_data_manager()
	: stream(std::make_shared<cuda_types::cuda_stream>()),
	  _vbo(0),
	  _ebo(0),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true),
	  amount_of_edges(sizeof(uint32_t), stream, true) {}

icosahedron_data_manager::icosahedron_data_manager(raw::UI vbo, raw::UI ebo,
												   std::shared_ptr<cuda_types::cuda_stream> stream)

	: stream(stream),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true),
	  amount_of_edges(sizeof(uint32_t), stream, true) {
	init(vbo, ebo);
}

void icosahedron_data_manager::init(raw::UI vbo, raw::UI ebo) {
	static int times_called = 0;
	// can be called only once in the lifetime
	assert(times_called == 0);
	_vbo = vbo;
	_ebo = ebo;
	vertices_handle =
		cuda_types::cuda_from_gl_data<raw::graphics::vertex>(&vertices_bytes, vbo, stream);
	indices_handle = cuda_types::cuda_from_gl_data<UI>(&indices_bytes, ebo, stream);

	vertices_second = cuda_types::cuda_buffer<raw::graphics::vertex>(vertices_bytes, stream, true);
	indices_second	= cuda_types::cuda_buffer<UI>(indices_bytes, stream, true);
	all_edges		= cuda_types::cuda_buffer<edge>(
		  predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(edge), stream, true);
	edge_to_vertex = cuda_types::cuda_buffer<uint32_t>(
		predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(uint32_t), stream, true);
	d_unique_edges = cuda_types::cuda_buffer<edge>(
		predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(edge), stream, true);
	amount_of_edges.zero_data(sizeof(uint32_t));

	inited = true;
	++times_called;
	cudaMemcpy(vertices_handle.get_data(), (void *)std::data(generate_icosahedron_vertices()),
			   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_handle.get_data(), (void *)std::data(generate_icosahedron_indices()),
			   num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice);
}

void icosahedron_data_manager::prepare(raw::UI vbo, raw::UI ebo) {
	if (!inited) {
		init(vbo, ebo);
		return;
	}
	vertices_handle.map();
	indices_handle.map();
	vertices_second.allocate(vertices_bytes);
	indices_second.allocate(indices_bytes);
	all_edges.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(edge));
	edge_to_vertex.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(uint32_t));
	d_unique_edges.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(edge));

	// Need to update func to also produce some cool data as tangent/bitangent
	cudaMemcpyAsync(vertices_handle.get_data(), (void *)std::data(generate_icosahedron_vertices()),
					num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice, stream->stream());
	cudaMemcpyAsync(indices_handle.get_data(), (void *)std::data(generate_icosahedron_indices()),
					num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice, stream->stream());
}
generation_context icosahedron_data_manager::create_context() {
	return generation_context {*this, _vbo, _ebo};
}

void icosahedron_data_manager::cleanup() {
	stream->sync();
	vertices_second.free();
	indices_second.free();
	vertices_handle.unmap();
	indices_handle.unmap();
	all_edges.free();
	d_unique_edges.free();
	edge_to_vertex.free();
	num_vertices_cpu  = 12;
	num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
}

// Icosahedron as most things in this project sounds horrifying, but, n-body (my algorithms), this
// thing, and tesselation are surprisingly easy things, just for some reason someone wanted give
// them scary names
constexpr std::array<graphics::vertex, 12>
icosahedron_data_manager::generate_icosahedron_vertices() {
	std::array<graphics::vertex, 12> vertices;
	int								 vertex_index = 0;

	const float unscaled_dist = std::sqrt(1.0f + GOLDEN_RATIO * GOLDEN_RATIO);
	const float scale		  = 1 / unscaled_dist;
	const float a			  = 1.0f * scale;
	const float b			  = GOLDEN_RATIO * scale;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			const auto sign1 = (j & 2) ? -1.0f : 1.0f;
			const auto sign2 = (j & 1) ? -1.0f : 1.0f;

			glm::vec3 point(1.0f);
			if (i == 0) {
				point = {sign1 * a, sign2 * b, 0.0f};
			} else if (i == 1) {
				point = {0.0f, sign1 * a, sign2 * b};
			} else {
				point = {sign1 * b, 0.0f, sign2 * a};
			}

			auto &v = vertices[vertex_index++];

			v.position = glm::normalize(point);
			v.normal   = v.position;

			constexpr glm::vec3 up = {0.0f, 1.0f, 0.0f};
			v.tangent			   = glm::normalize(glm::cross(up, v.normal));
			v.bitangent			   = glm::normalize(glm::cross(v.normal, v.tangent));
			v.tex_coord.x		   = 0.5f + std::atan2(v.normal.z, v.normal.x) / (2.0f * PI);
			v.tex_coord.y		   = 0.5f - std::asin(v.normal.y) / PI;
		}
	}
	return vertices;
}

constexpr std::array<UI, 60> icosahedron_data_manager::generate_icosahedron_indices() {
	return {2, 10, 4,  2, 4,  0, 2, 0, 5, 2, 5,	 11, 2, 11, 10, 0, 4, 8, 4, 10,
			6, 10, 11, 3, 11, 5, 7, 5, 0, 9, 1,	 8,	 6, 1,	6,	3, 1, 3, 7, 1,
			7, 9,  1,  9, 8,  6, 8, 4, 3, 6, 10, 7,	 3, 11, 9,	7, 5, 8, 9, 0};
}

} // namespace raw::sphere_generation