//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_KERNEL_LAUNCHER_H
#define SPACE_EXPLORER_KERNEL_LAUNCHER_H
#include "cuda_types/fwd.h"
namespace raw {
    extern void launch_tessellation(const raw::vertex *in_vertices, const UI *in_indices,
                                    raw::vertex *out_vertices, UI *out_indices,
                                    uint32_t *p_vertex_count, uint32_t *p_triangle_count,
                                    size_t num_input_triangles, cuda_stream &stream);

    extern void launch_orthogonalization(raw::vertex *vertices, size_t num_vertices, cuda_stream &stream);
}

#endif // SPACE_EXPLORER_KERNEL_LAUNCHER_H