//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_KERNEL_LAUNCHER_H
#define SPACE_EXPLORER_KERNEL_LAUNCHER_H
#include <cuda_runtime.h>
#include "sphere_generation/fwd.h"

/**
 * Launch GPU tessellation kernels to subdivide and generate sphere geometry.
 *
 * This schedules CUDA work on the provided stream to process input vertex/index buffers
 * and produce tessellated output vertex/index buffers. Input pointers and output pointers
 * refer to device memory. The function updates the counters pointed to by p_vertex_count,
 * p_triangle_count, and p_edges_cound to reflect the number of vertices, triangles, and
 * edges produced/consumed.
 *
 * @param in_vertices Device pointer to input vertices.
 * @param in_indices Device pointer to input indices.
 * @param all_edges Device pointer to the edge list used during tessellation.
 * @param out_vertices Device pointer where output vertices will be written.
 * @param out_indices Device pointer where output indices will be written.
 * @param d_unique_edges Device pointer used to store unique edges discovered during processing.
 * @param edge_to_vertex Device pointer mapping edges to vertex indices.
 * @param p_vertex_count Device pointer to a uint32_t containing the current vertex count; updated with the final count.
 * @param p_triangle_count Device pointer to a uint32_t containing the current triangle count; updated with the final count.
 * @param p_edges_cound Device pointer to a uint32_t containing the current edge count; updated with the final count.
 * @param stream CUDA stream on which kernels and memory operations are enqueued.
 * @param steps Number of tessellation/subdivision iterations to perform.
 */
namespace raw::sphere_generation {
extern void launch_tessellation(raw::graphics::vertex *in_vertices, UI *in_indices, edge *all_edges,
								raw::graphics::vertex *out_vertices, UI *out_indices,
								edge *d_unique_edges, uint32_t *edge_to_vertex,
								uint32_t *p_vertex_count, uint32_t *p_triangle_count,
								uint32_t *p_edges_cound, cudaStream_t &stream, uint32_t steps);

} // namespace raw::sphere_generation

#endif // SPACE_EXPLORER_KERNEL_LAUNCHER_H