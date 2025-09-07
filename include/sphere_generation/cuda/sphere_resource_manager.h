//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include <raw_memory.h>

#include <array>
#include <glm/glm.hpp>
// clang-format off

#include "../../common/scoped_resource_handle.h"
#include "sphere_generation/i_sphere_resource_manager.h"
#include "device_types/cuda/buffer.h"
#include "device_types/cuda/from_gl/buffer.h"
#include "graphics/vertex.h"
#include "sphere_generation/cuda/fwd.h"

// clang-format on
namespace raw::sphere_generation::cuda {
using namespace device_types::cuda;
using cuda_tessellation_data =
	std::tuple<graphics::vertex*, unsigned*, edge*, graphics::vertex*, unsigned*, edge*, unsigned*,
			   unsigned*, unsigned*, unsigned*>;
// this class serves as a nice thing to warp around hard things in generating sphere from
// icosahedron
class sphere_resource_manager : public i_sphere_resource_manager {
private:
	cuda::from_gl::buffer<raw::graphics::vertex> vertices_handle;
	cuda::from_gl::buffer<UI>					 indices_handle;
	std::shared_ptr<cuda::cuda_stream>			 stream;

	UI _vbo;
	UI _ebo;

	cuda::buffer<raw::graphics::vertex> vertices_second;
	cuda::buffer<UI>					indices_second;

	cuda::buffer<uint32_t> amount_of_triangles;
	cuda::buffer<uint32_t> amount_of_vertices;
	cuda::buffer<uint32_t> amount_of_edges;

	cuda::buffer<edge>	   all_edges;
	cuda::buffer<edge>	   d_unique_edges;
	cuda::buffer<uint32_t> edge_to_vertex;

	size_t vertices_bytes = 0;
	size_t indices_bytes  = 0;

	uint32_t num_vertices_cpu  = 12;
	uint32_t num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;

	bool inited = false;

	friend generation_context;
	// Called once when the object is created (or generate function called first time)
	void init(UI vbo, UI ebo);

	// requires to be called each time before acquiring the data
	void prepare(UI vbo, UI ebo) override;

	// requires to be called each time after generation complete
	void cleanup() override;

public:
	sphere_resource_manager();

	sphere_resource_manager(UI vbo, UI ebo, std::shared_ptr<cuda::cuda_stream> stream);

	generation_context create_context() override;

	[[nodiscard]] tessellation_data get_data() const override {
		return std::make_tuple(
			device_ptr(vertices_handle.get_data()), device_ptr(indices_handle.get_data()),
			device_ptr(all_edges.get()), device_ptr(vertices_second.get()),
			device_ptr(indices_second.get()), device_ptr(d_unique_edges.get()),
			device_ptr(edge_to_vertex.get()), device_ptr(amount_of_vertices.get()),
			device_ptr(amount_of_triangles.get()), device_ptr(amount_of_edges.get()));
	}
};
} // namespace raw::sphere_generation::cuda
#endif // SPACE_EXPLORER_MESH_GENERATOR_H