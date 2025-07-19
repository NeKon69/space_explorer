//
// Created by progamers on 7/18/25.
//
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tesselation_kernel.h"
namespace raw {
void launch_tesselation(const glm::vec3* in_vertices, const UI* in_indices, glm::vec3* out_vertices,
						UI* out_indices, uint32_t* p_vertex_count, uint32_t* p_triangle_count,
						size_t num_input_triangles, float radius) {
	constexpr auto threads_per_block = 256;
	auto		   blocks			 = (num_input_triangles + 256 - 1) / 256;
	subdivide<<<blocks, threads_per_block>>>(in_vertices, in_indices, out_vertices, out_indices,
											 p_vertex_count, p_triangle_count, num_input_triangles,
											 radius);
}
} // namespace raw