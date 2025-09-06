//
// Created by progamers on 7/7/25.
//

#include "sphere_generation/cuda/icosahedron_data_manager.h"

#include <numbers>

#include "core/clock.h"
#include "cuda_types/buffer.h"
#include "sphere_generation/cuda/kernel_launcher.h"

namespace raw::sphere_generation::cuda {

sphere_resource_manager::sphere_resource_manager()
	: stream(std::make_shared<cuda_types::cuda_stream>()),
	  _vbo(0),
	  _ebo(0),
	  amount_of_triangles(sizeof(uint32_t), stream),
	  amount_of_vertices(sizeof(uint32_t), stream),
	  amount_of_edges(sizeof(uint32_t), stream) {}

sphere_resource_manager::sphere_resource_manager(raw::UI vbo, raw::UI ebo,
												 std::shared_ptr<cuda_types::cuda_stream> stream)

	: stream(stream),
	  amount_of_triangles(sizeof(uint32_t), stream),
	  amount_of_vertices(sizeof(uint32_t), stream),
	  amount_of_edges(sizeof(uint32_t), stream) {
	init(vbo, ebo);
}

void sphere_resource_manager::init(raw::UI vbo, raw::UI ebo) {
	static int times_called = 0;
	// can be called only once in the lifetime
	assert(times_called == 0);
	_vbo			= vbo;
	_ebo			= ebo;
	vertices_handle = cuda_types::from_gl::buffer<graphics::vertex>(&vertices_bytes, vbo, stream);
	indices_handle	= cuda_types::from_gl::buffer<UI>(&indices_bytes, ebo, stream);

	vertices_second.set_stream(stream);
	indices_second.set_stream(stream);
	all_edges.set_stream(stream);
	edge_to_vertex.set_stream(stream);
	d_unique_edges.set_stream(stream);
	amount_of_edges.zero_data(sizeof(uint32_t));

	inited = true;
	++times_called;
}

void sphere_resource_manager::prepare(raw::UI vbo, raw::UI ebo) {
	if (!inited) {
		init(vbo, ebo);
		return;
	}
	vertices_handle.map();
	indices_handle.map();
	vertices_second.allocate(vertices_bytes);
	indices_second.allocate(indices_bytes);
	all_edges.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3 * sizeof(edge));
	edge_to_vertex.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * sizeof(uint32_t));
	d_unique_edges.allocate(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * sizeof(edge));
}
generation_context sphere_resource_manager::create_context() {
	return generation_context {this, _vbo, _ebo};
}

void sphere_resource_manager::cleanup() {
	vertices_handle.unmap();
	indices_handle.unmap();
	vertices_second.free();
	indices_second.free();
	all_edges.free();
	d_unique_edges.free();
	edge_to_vertex.free();
	num_vertices_cpu  = 12;
	num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
}

} // namespace raw::sphere_generation::cuda