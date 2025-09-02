//
// Created by progamers on 7/18/25.
//
#include <thrust/sort.h>
#include <thrust/system/cuda/memory_resource.h>

#include "cuda_types/buffer.h"
#include "cuda_types/error.h"
#include "cuda_types/stream.h"
#include "graphics/vertex.h"
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tessellation_kernel.h"

namespace raw::sphere_generation {
/**
 * @brief Strict ordering comparator for edges (ascending by endpoints).
 *
 * Compares two edges first by their v0 index, and if those are equal, by their v1 index.
 * Designed to be usable on both host and device for sorting operations (e.g., with Thrust).
 *
 * @param a First edge to compare.
 * @param b Second edge to compare.
 * @return true if edge `a` should come before edge `b` (a < b) under the ordering.
 */
__device__ __host__ bool edge_comparator(const edge &a, const edge &b) {
	if (a.v0 > b.v0)
		return false;
	if (a.v0 < b.v0)
		return true;
	return a.v1 < b.v1;
}
/**
 * @brief Perform iterative triangular mesh tessellation (subdivision) on the device.
 *
 * Performs `steps` iterations of midpoint subdivision: for each iteration it generates edges
 * from the current triangle index buffer, creates unique midpoint vertices for edges,
 * rebuilds the triangle index buffer (each triangle becomes 4), and updates device-side
 * vertex/triangle counts. After the loop it computes per-vertex tangents/bitangents/UVs.
 *
 * The function operates entirely on device memory and CUDA streams: it launches device
 * kernels, performs device-to-device and device-to-host count transfers, and synchronizes
 * the provided CUDA stream at key points. If `steps` is odd, the final vertex/index buffers
 * are copied back into the original input buffers so the caller's base buffers contain the
 * final tessellated mesh.
 *
 * @param in_vertices Device pointer to the current input vertex buffer (may be swapped internally).
 * @param in_indices Device pointer to the current input index buffer (may be swapped internally).
 * @param all_edges Device array used to store generated edges for the current iteration.
 * @param out_vertices Device pointer to the output vertex buffer (used as a swap target).
 * @param out_indices Device pointer to the output index buffer (used as a swap target).
 * @param d_unique_edges Device array used to store unique edges identified for midpoint creation.
 * @param edge_to_vertex Device array mapping an edge key to its midpoint vertex index.
 * @param p_vertex_count Device pointer to a uint32_t counter holding the current vertex count;
 *        updated on-device as midpoints are created. (Device-side writable.)
 * @param p_triangle_count Device pointer to a uint32_t counter holding the current triangle count;
 *        the function multiplies this count by 4 each subdivision iteration. (Device-side writable.)
 * @param p_unique_edges_count Device pointer to a uint32_t used to return the number of unique edges
 *        discovered in each iteration; the host-side copy of this value is used to size subsequent sorts.
 * @param stream CUDA stream on which all operations and kernel launches are enqueued.
 * @param steps Number of subdivision iterations to perform.
 */
void launch_tessellation(raw::graphics::vertex *in_vertices, UI *in_indices, edge *all_edges,

						 raw::graphics::vertex *out_vertices, UI *out_indices, edge *d_unique_edges,
						 uint32_t *edge_to_vertex, uint32_t *p_vertex_count,
						 uint32_t *p_triangle_count, uint32_t *p_unique_edges_count,
						 cudaStream_t &stream, uint32_t steps) {
	auto		   base_in_vertices	 = in_vertices;
	const auto	   base_in_indices	 = in_indices;
	uint32_t	   num_vertices_cpu	 = 12;
	constexpr auto threads_per_block = 1024;

	cuda_types::cuda_buffer<uint32_t, cuda_types::side::host> num_triangles_cpu(sizeof(uint32_t));
	cuda_types::cuda_buffer<uint32_t, cuda_types::side::host> num_unique_edges_cpu(
		sizeof(uint32_t));
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
			CUDA_SAFE_CALL(cudaMemcpyAsync(out_vertices, in_vertices,
										   12 * sizeof(raw::graphics::vertex),
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

		CUDA_SAFE_CALL(cudaMemcpyAsync(num_unique_edges_cpu.get(), p_unique_edges_count,
									   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
		thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), d_unique_edges,
							d_unique_edges + *num_unique_edges_cpu, edge_to_vertex);

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
	cudaMemcpyAsync(&num_vertices_cpu, p_vertex_count, sizeof(uint32_t), cudaMemcpyDeviceToHost,
					stream);
	cudaStreamSynchronize(stream);
	if (steps % 2 != 0) {
		cudaMemcpyAsync(base_in_vertices, out_vertices, num_vertices_cpu * sizeof(graphics::vertex),
						cudaMemcpyDeviceToDevice, stream);
		cudaMemcpyAsync(base_in_indices, out_indices, *num_triangles_cpu * 3 * sizeof(UI),
						cudaMemcpyDeviceToDevice, stream);
	}
	blocks = (num_vertices_cpu + threads_per_block - 1) / threads_per_block;

	calculate_tbn_and_uv<<<blocks, threads_per_block, 0, stream>>>(in_vertices, num_vertices_cpu);
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

} // namespace raw::sphere_generation