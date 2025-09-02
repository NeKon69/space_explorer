//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define GLM_CUDA_FORCE_DEVICE_FUNC

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * @brief Convert an edge to a canonical representation.
 *
 * Ensures the edge's stored vertex indices are ordered and associated metadata
 * (if any) are adjusted so the same undirected edge maps to a single canonical
 * representation.
 *
 * @param edge Edge object to normalize in-place.
 * @param i0 One vertex index of the edge (used to determine canonical ordering).
 * @param i1 The other vertex index of the edge (used to determine canonical ordering).
 */

/**
 * @brief Generate edge records for all input triangles.
 *
 * Converts triangle index triples from `in_indices` into three edge entries per
 * triangle and writes them to `out_edges`.
 *
 * @param in_indices Pointer to input triangle indices (3 * num_input_triangles entries).
 * @param out_edges Output buffer for generated edges (3 * num_input_triangles entries).
 * @param num_input_triangles Number of input triangles to process.
 */

/**
 * @brief Create unique midpoint vertices for a set of edges.
 *
 * For each edge in `sorted_edges`, creates (when needed) a midpoint vertex in
 * `out_vertices` and produces mappings that link each unique edge to its
 * midpoint vertex index. Updates `p_vertex_count` with the next free vertex
 * index and writes the set of unique edges into `unique_edges`.
 *
 * @param sorted_edges Input edges sorted so duplicate edges are adjacent.
 * @param in_vertices Original vertex array used to compute midpoints.
 * @param out_vertices Output vertex array; midpoint vertices are appended here.
 * @param p_vertex_count Pointer to a running vertex count; incremented as midpoints are added.
 * @param unique_edges Output array to receive the list of unique edges discovered.
 * @param edge_to_vertex Output array mapping edge indices to the corresponding midpoint vertex index.
 * @param num_unique_edges Output pointer receiving the number of unique edges written.
 * @param num_total_edges Total number of edges in `sorted_edges`.
 */

/**
 * @brief Find a target edge in an array of unique edges.
 *
 * Performs a linear search for `target` within `unique_edges` and returns its
 * index if found.
 *
 * @param unique_edges Array of unique edges to search.
 * @param num_unique_edges Number of entries in `unique_edges`.
 * @param target Edge to locate.
 * @return Index of the matching edge in `unique_edges`, or -1 if not found.
 */

/**
 * @brief Build subdivided triangle indices using edge midpoint mappings.
 *
 * Using the original `in_indices` and the mapping from unique edges to midpoint
 * vertices (`edge_to_vertex`), writes the indices for subdivided triangles to
 * `out_indices`.
 *
 * @param in_indices Input triangle indices (3 * num_input_triangles entries).
 * @param out_indices Output triangle indices after subdivision (expected size 12 * num_input_triangles).
 * @param unique_edges Array of unique edges corresponding to `edge_to_vertex`.
 * @param edge_to_vertex Mapping from unique edge index to midpoint vertex index.
 * @param p_num_unique_edges Pointer to the number of unique edges available.
 * @param num_input_triangles Number of input triangles to process.
 */

/**
 * @brief Compute tangent, bitangent, normal (TBN) and UVs for vertices.
 *
 * For each vertex in `vertices[0..num_input_vertices)`, computes/updates the
 * normal, tangent, bitangent (TBN) and UV coordinates based on the vertex
 * attributes present.
 *
 * @param vertices Array of vertices to process in-place.
 * @param num_input_vertices Number of vertices in `vertices`.
 */

/**
 * @brief Subdivide the mesh by inserting midpoint vertices and producing new triangles.
 *
 * Reads `in_vertices` and `in_indices`, generates midpoint vertices and new
 * triangle indices, appends vertices into `out_vertices` and indices into
 * `out_indices`, and updates the provided counts.
 *
 * @param in_vertices Input vertex array.
 * @param in_indices Input triangle index array (3 * num_input_triangles entries).
 * @param out_vertices Output vertex array; new vertices are appended.
 * @param out_indices Output triangle indices after subdivision.
 * @param p_vertex_count Pointer to a running vertex count; updated as new vertices are appended.
 * @param p_triangle_count Pointer to a running triangle count; updated with the number of output triangles.
 * @param num_input_triangles Number of input triangles to subdivide.
 */

/**
 * @brief Orthogonalize vertex tangent frames.
 *
 * Adjusts vertex tangent/bitangent/normal vectors in-place so they form an
 * orthonormal basis for each vertex in `vertices[0..vertex_count)`.
 *
 * @param vertices Array of vertices whose tangent frames will be orthogonalized.
 * @param vertex_count Number of vertices to process.
 */
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
extern __global__ void calculate_tbn_and_uv(raw::graphics::vertex *vertices,
											uint32_t			   num_input_vertices);

extern __global__ void subdivide(raw::graphics::vertex *in_vertices, unsigned int *in_indices,
								 raw::graphics::vertex *out_vertices, unsigned int *out_indices,
								 uint32_t *p_vertex_count, uint32_t *p_triangle_count,
								 size_t num_input_triangles);

extern __global__ void orthogonalize(raw::graphics::vertex *vertices, uint32_t vertex_count);
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_TESSELLATION_KERNEL_H