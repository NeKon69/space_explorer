//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include <raw_memory.h>

#include <array>
#include <glm/glm.hpp>
// clang-format off

#include "sphere_generation/generation_context.h"
#include "sphere_generation/i_sphere_resource_manager.h"
#include "cuda_types/buffer.h"
#include "cuda_types/from_gl/buffer.h"
#include "graphics/vertex.h"
#include "sphere_generation/cuda/fwd.h"

// clang-format on
namespace raw::sphere_generation::cuda {
using cuda_tessellation_data =
	std::tuple<graphics::vertex*, unsigned*, edge*, graphics::vertex*, unsigned*, edge*, unsigned*,
			   unsigned*, unsigned*, unsigned*>;
// this class serves as a nice thing to warp around hard things in generating sphere from
// icosahedron
class sphere_resource_manager : public i_sphere_resource_manager {
private:
	cuda_types::from_gl::buffer<raw::graphics::vertex> vertices_handle;
	cuda_types::from_gl::buffer<UI>					   indices_handle;
	std::shared_ptr<cuda_types::cuda_stream>		   stream;

	UI _vbo;
	UI _ebo;

	cuda_types::cuda_buffer<raw::graphics::vertex> vertices_second;
	cuda_types::cuda_buffer<UI>					   indices_second;

	cuda_types::cuda_buffer<uint32_t> amount_of_triangles;
	cuda_types::cuda_buffer<uint32_t> amount_of_vertices;
	cuda_types::cuda_buffer<uint32_t> amount_of_edges;

	cuda_types::cuda_buffer<edge>	  all_edges;
	cuda_types::cuda_buffer<edge>	  d_unique_edges;
	cuda_types::cuda_buffer<uint32_t> edge_to_vertex;

	size_t vertices_bytes = 0;
	size_t indices_bytes  = 0;

	uint32_t num_vertices_cpu  = 12;
	uint32_t num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;

	bool inited = false;

	friend class sphere_generation::generation_context;
	// Called once when the object is created (or generate function called first time)
	void init(UI vbo, UI ebo);

	// requires to be called each time before acquiring the data
	void prepare(UI vbo, UI ebo) override;

	// requires to be called each time after generation complete
	void cleanup() override;

public:
	sphere_resource_manager();

	sphere_resource_manager(UI vbo, UI ebo, std::shared_ptr<cuda_types::cuda_stream> stream);

	generation_context create_context() override;

	static constexpr std::array<graphics::vertex, 12> generate_icosahedron_vertices();

	static constexpr std::array<UI, 60> generate_icosahedron_indices();

	[[nodiscard]] tessellation_data get_data() const override {
		return std::make_tuple(vertices_handle.get_data(), indices_handle.get_data(),
							   static_cast<edge_base*>(all_edges.get()), vertices_second.get(),
							   indices_second.get(), static_cast<edge_base*>(d_unique_edges.get()),
							   edge_to_vertex.get(), amount_of_vertices.get(),
							   amount_of_triangles.get(), amount_of_edges.get());
	}
};
} // namespace raw::sphere_generation::cuda
#endif // SPACE_EXPLORER_MESH_GENERATOR_H