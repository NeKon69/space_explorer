//
// Created by progamers on 7/18/25.
//
#include <thrust/sort.h>

#include "device_types/cuda/buffer.h"
#include "device_types/cuda/error.h"
#include "graphics/vertex.h"
#include "sphere_generation/basic_geomerty.h"
#include "sphere_generation/cuda/tessellation_kernel.h"
#include "sphere_generation/cuda/tessellation_launcher.h"

namespace raw::sphere_generation::cuda {
using namespace device_types::cuda;
void launch_tessellation(raw::graphics::vertex *in_vertices, UI *in_indices, edge *all_edges,
						 raw::graphics::vertex *out_vertices, UI *out_indices, edge *d_unique_edges,
						 uint32_t *edge_to_vertex, uint32_t *p_vertex_count,
						 uint32_t *p_triangle_count, uint32_t *p_unique_edges_count,
						 cudaStream_t &stream, uint32_t steps) {
	cudaMemcpyAsync(in_vertices, std::data(generate_icosahedron_vertices()),
					sizeof(graphics::vertex) * 12, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(in_indices, std::data(generate_icosahedron_indices()), sizeof(uint32_t) * 60,
					cudaMemcpyHostToDevice);


	auto		   base_in_vertices	 = in_vertices;
	const auto	   base_in_indices	 = in_indices;
	constexpr auto threads_per_block = 1024;

	cuda::buffer<uint32_t, cuda::side::host> num_triangles_cpu(sizeof(uint32_t));
	cuda::buffer<uint32_t, cuda::side::host> num_unique_edges_cpu(sizeof(uint32_t));
	*num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
	auto blocks		   = (*num_triangles_cpu + threads_per_block - 1) / threads_per_block;
	int	 base		   = 12;
	CUDA_SAFE_CALL(
		cudaMemcpyAsync(p_vertex_count, &base, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
	// This shit right here, took me fckn 2 hours to understand LOL
	for (uint32_t i = 0; i < steps; ++i) {
		// we need to sync here for above operation to finish, get correct "num_vertices_cpu" and
		// then perform bottom operation with correct byte size
		if (i != 0) {
			CUDA_SAFE_CALL(
				cudaMemcpyAsync(out_vertices, in_vertices,
								predef::MAXIMUM_AMOUNT_OF_VERTICES * sizeof(raw::graphics::vertex),
								cudaMemcpyDeviceToDevice, stream));
		} else {
			// We know that at the first iteration we have the least amount of vertices, so it's
			// just a small optimization
			CUDA_SAFE_CALL(
				cudaMemcpyAsync(out_vertices, in_vertices,
								predef::BASIC_AMOUNT_OF_VERTICES * sizeof(raw::graphics::vertex),
								cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_SAFE_CALL(cudaMemsetAsync(p_unique_edges_count, 0, sizeof(uint32_t), stream));

		blocks = (*num_triangles_cpu + threads_per_block - 1) / threads_per_block;
		generate_edges<<<blocks, threads_per_block, 0, stream>>>(in_indices, all_edges,
																 *num_triangles_cpu);
		thrust::sort(thrust::cuda::par_nosync.on(stream), all_edges,
					 all_edges + *num_triangles_cpu * 3);

		blocks = (*num_triangles_cpu * 3 + threads_per_block - 1) / threads_per_block;
		create_unique_midpoint_vertices<<<blocks, threads_per_block, 0, stream>>>(
			all_edges, in_vertices, out_vertices, p_vertex_count, d_unique_edges, edge_to_vertex,
			p_unique_edges_count, *num_triangles_cpu * 3);

		sort_by_key<<<1, 1, 0, stream>>>(d_unique_edges, p_unique_edges_count, edge_to_vertex);

		blocks = (*num_triangles_cpu + threads_per_block - 1) / threads_per_block;

		create_triangles<<<blocks, threads_per_block, 0, stream>>>(
			in_indices, out_indices, d_unique_edges, edge_to_vertex, p_unique_edges_count,
			*num_triangles_cpu);

		*num_triangles_cpu *= 4;
		std::swap(in_vertices, out_vertices);
		std::swap(in_indices, out_indices);
	}
	in_vertices = base_in_vertices;
	in_indices	= base_in_indices;
	// we can predict how many vertices we have since it's the last step
	// we can probably predict amount of vertices at each step, but I am too lazy to test that
	uint32_t amount_of_vertices = 10u * (1u << (2u * steps)) + 2u;
	if (steps % 2 != 0) {
		cudaMemcpyAsync(base_in_vertices, out_vertices,
						amount_of_vertices * sizeof(graphics::vertex), cudaMemcpyDeviceToDevice,
						stream);
		cudaMemcpyAsync(base_in_indices, out_indices, *num_triangles_cpu * 3 * sizeof(UI),
						cudaMemcpyDeviceToDevice, stream);
	}
	blocks = (amount_of_vertices + threads_per_block - 1) / threads_per_block;
	// But just for safe measure if we are wrong pass the actual value
	calculate_tbn_and_uv<<<blocks, threads_per_block, 0, stream>>>(in_vertices, p_vertex_count);
}

} // namespace raw::sphere_generation::cuda