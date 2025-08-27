//
// Created by progamers on 7/18/25.
//
#include <thrust/sort.h>

#include "../../include/graphics/vertex.h"
#include "cuda_types/stream.h"
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tessellation_kernel.h"

namespace raw::sphere_generation {
void launch_tessellation(raw::graphics::vertex *in_vertices, UI *in_indices, edge *all_edges,
						 raw::graphics::vertex *out_vertices, UI *out_indices, edge *d_unique_edges,
						 uint32_t *edge_to_vertex, uint32_t *p_vertex_count,
						 uint32_t *p_triangle_count, uint32_t *p_unique_edges_count,
						 cudaStream_t &stream, uint32_t steps) {
	const auto	   base_in_vertices	 = in_vertices;
	const auto	   base_in_indices	 = in_indices;
	uint32_t	   num_vertices_cpu	 = 12;
	uint32_t	   num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
	constexpr auto threads_per_block = 1024;
	auto		   blocks = (num_triangles_cpu + threads_per_block - 1) / threads_per_block;
	// This shit right here, took me fckn 2 hours to understand LOL
	for (uint32_t i = 0; i < steps; ++i) {
		cudaMemcpyAsync(p_vertex_count, &num_vertices_cpu, sizeof(num_vertices_cpu),
						cudaMemcpyHostToDevice, stream);

		in_vertices = base_in_vertices == in_vertices && i % 2 != 0 ? out_vertices : in_vertices;
		in_indices	= base_in_indices == in_indices && i % 2 != 0 ? out_indices : in_indices;

		generate_edges<<<blocks, threads_per_block, 0, stream>>>(in_indices, all_edges,
																 num_triangles_cpu);

		thrust::sort(thrust::device, all_edges, all_edges + num_triangles_cpu * 3, edge());

		blocks = (num_triangles_cpu * 3 + threads_per_block - 1) / threads_per_block;
		create_unique_midpoint_vertices<<<blocks, threads_per_block, 0, stream>>>(
			all_edges, in_vertices, p_vertex_count, d_unique_edges, edge_to_vertex,
			p_unique_edges_count, num_triangles_cpu * 3);

		create_triangles<<<blocks, threads_per_block, 0, stream>>>(
			in_indices, out_indices, d_unique_edges, edge_to_vertex, p_unique_edges_count,
			num_triangles_cpu);

		num_vertices_cpu += 3 * num_triangles_cpu;
		num_triangles_cpu *= 4;
		blocks = (num_triangles_cpu + threads_per_block - 1) / threads_per_block;
	}
	in_vertices = base_in_vertices;
	in_indices	= base_in_indices;
	if (steps % 2 != 0) {
		cudaMemcpyAsync(in_vertices, out_vertices, num_vertices_cpu * sizeof(glm::vec3),
						cudaMemcpyDeviceToDevice, stream);
	}
	calculate_tbn_and_uv<<<blocks * 3, threads_per_block, 0, stream>>>(in_vertices, num_vertices_cpu);
}

void launch_orthogonalization(raw::graphics::vertex *vertices, size_t num_vertices,
							  cudaStream_t &stream) {
	constexpr auto threads_per_block = 256;
	auto		   blocks			 = (num_vertices + threads_per_block - 1) / 256;
	orthogonalize<<<blocks, threads_per_block, 0, stream>>>(vertices, num_vertices);
}
} // namespace raw::sphere_generation