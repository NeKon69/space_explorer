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
__device__ __host__ bool edge_comparator(const edge &a, const edge &b) {
	if (a.v0 > b.v0)
		return false;
	if (a.v0 < b.v0)
		return true;
	return a.v1 < b.v1;
}
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
	*num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
	auto blocks		   = (*num_triangles_cpu + threads_per_block - 1) / threads_per_block;
	// This shit right here, took me fckn 2 hours to understand LOL
	printf("[Host] About to launch generate_edges. out_edges pointer is: %p\n", all_edges);

	generate_edges<<<blocks, threads_per_block, 0, stream>>>(in_indices, all_edges,
															 *num_triangles_cpu);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	for (uint32_t i = 0; i < steps; ++i) {
		// we need to sync here for above operation to finish, get correct "num_vertices_cpu" and
		// then perform bottom operation with correct byte size
		if (i != 0) {
			CUDA_SAFE_CALL(
				cudaMemcpyAsync(out_vertices, in_vertices,
								predef::MAXIMUM_AMOUNT_OF_VERTICES * sizeof(raw::graphics::vertex),
								cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_SAFE_CALL(cudaMemsetAsync(p_unique_edges_count, 0, sizeof(uint32_t), stream));

		blocks = (*num_triangles_cpu + threads_per_block - 1) / threads_per_block;
		generate_edges<<<blocks, threads_per_block, 0, stream>>>(in_indices, all_edges,
																 *num_triangles_cpu);
		void *p_test_allocation = nullptr;
		CUDA_SAFE_CALL(cudaMalloc(&p_test_allocation, 16));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaFree(p_test_allocation));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		std::cout << "--- Subdivision Step " << i << " ---" << std::endl;
		size_t num_edges_to_sort  = (size_t)*num_triangles_cpu * 3;
		size_t all_edges_capacity = predef::MAXIMUM_AMOUNT_OF_TRIANGLES * 3;

		std::cout << "Number of triangles (input): " << *num_triangles_cpu << std::endl;
		std::cout << "Number of edges to sort:     " << num_edges_to_sort << std::endl;
		std::cout << "Capacity of all_edges buffer:  " << all_edges_capacity << std::endl;

		if (num_edges_to_sort > all_edges_capacity) {
			std::cerr << "FATAL ERROR: Attempting to sort more edges than allocated!" << std::endl;
		}
		thrust::sort(thrust::cuda::par_nosync.on(stream), all_edges,
					 all_edges + *num_triangles_cpu * 3);

		blocks = (*num_triangles_cpu * 3 + threads_per_block - 1) / threads_per_block;
		create_unique_midpoint_vertices<<<blocks, threads_per_block, 0, stream>>>(
			all_edges, in_vertices, out_vertices, p_vertex_count, d_unique_edges, edge_to_vertex,
			p_unique_edges_count, *num_triangles_cpu * 3);

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

	// calculate_tbn_and_uv<<<blocks, threads_per_block, 0, stream>>>(in_vertices,
	// num_vertices_cpu);
	cudaStreamSynchronize(stream);
}

void launch_orthogonalization(raw::graphics::vertex *vertices, size_t num_vertices,
							  cudaStream_t &stream) {
	constexpr auto threads_per_block = 256;
	auto		   blocks			 = (num_vertices + threads_per_block - 1) / 256;
	orthogonalize<<<blocks, threads_per_block, 0, stream>>>(vertices, num_vertices);
}

__global__ void dummy_kernel(int *counter) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		atomicAdd(counter, 1);
	}
}

// A simple C++ function to launch the dummy kernel
void launch_dummy_kernel() {
	int *d_counter;
	int	 h_counter = 0;

	// Allocate memory on the GPU
	cudaMalloc(&d_counter, sizeof(int));

	// Copy initial value (0) to GPU
	cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

	// Launch the kernel
	std::cout << "[Debug] Launching dummy kernel..." << std::endl;
	dummy_kernel<<<1, 1>>>(d_counter);

	// Check for any launch errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "[Debug] Dummy kernel launch failed: " << cudaGetErrorString(err) << std::endl;
	} else {
		std::cout << "[Debug] Dummy kernel launched successfully." << std::endl;
	}

	// Synchronize to ensure the kernel has finished
	cudaDeviceSynchronize();

	// Copy the result back from GPU
	cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_counter);

	// Print the result
	std::cout << "[Debug] Dummy kernel result: " << h_counter << " (should be 1)" << std::endl;
}

} // namespace raw::sphere_generation