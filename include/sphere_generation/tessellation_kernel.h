//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define GLM_CUDA_FORCE_DEVICE_FUNC

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace raw::sphere_generation {
extern __device__ void make_canonical_edge(edge &edge, uint32_t i0, uint32_t i1);

extern __global__ void generate_edges(const UI *in_indices, edge *out_edges,
									  size_t num_input_triangles);
extern __global__ void create_unique_midpoint_vertices(
	const edge *sorted_edges, const graphics::vertex *in_vertices, graphics::vertex *out_vertices,
	uint32_t *p_vertex_count, edge *unique_edges, uint32_t *edge_to_vertex,
	uint32_t *num_unique_edges, size_t num_total_edges);
extern __device__ int find_edge(const edge *unique_edges, uint32_t num_unique_edges, edge target);

extern __global__ void create_triangles(const UI *in_indices, UI *out_indices,
										const edge *unique_edges, const uint32_t *edge_to_vertex,
										const uint32_t *p_num_unique_edges,
										const size_t	num_input_triangles);

extern __global__ void subdivide(raw::graphics::vertex *in_vertices, unsigned int *in_indices,
								 raw::graphics::vertex *out_vertices, unsigned int *out_indices,
								 uint32_t *p_vertex_count, uint32_t *p_triangle_count,
								 size_t num_input_triangles);

extern __global__ void orthogonalize(raw::graphics::vertex *vertices, uint32_t vertex_count);
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_TESSELLATION_KERNEL_H