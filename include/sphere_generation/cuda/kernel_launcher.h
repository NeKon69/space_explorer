//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_KERNEL_LAUNCHER_H
#define SPACE_EXPLORER_KERNEL_LAUNCHER_H
#include <cuda_runtime.h>
#include "fwd.h"

namespace raw::sphere_generation::cuda {
extern void launch_tessellation(raw::graphics::vertex *in_vertices, UI *in_indices, edge *all_edges,
								raw::graphics::vertex *out_vertices, UI *out_indices,
								edge *d_unique_edges, uint32_t *edge_to_vertex,
								uint32_t *p_vertex_count, uint32_t *p_triangle_count,
								uint32_t *p_edges_cound, cudaStream_t &stream, uint32_t steps);

} // namespace raw::sphere_generation

#endif // SPACE_EXPLORER_KERNEL_LAUNCHER_H