//
// Created by progamers on 7/18/25.
//
#include "../../include/graphics/vertex.h"
#include "cuda_types/stream.h"
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tessellation_kernel.h"

namespace raw::sphere_generation {
    void launch_tessellation(const raw::vertex *in_vertices, const UI *in_indices,
                             raw::vertex *out_vertices, UI *out_indices, uint32_t *p_vertex_count,
                             uint32_t *p_triangle_count, size_t num_input_triangles,
                             raw::cuda_types::cuda_stream &stream) {
        constexpr auto threads_per_block = 256;
        auto blocks = (num_input_triangles + threads_per_block - 1) / 256;
        subdivide<<<blocks, threads_per_block, 0, stream.stream()>>>(
            in_vertices, in_indices, out_vertices, out_indices, p_vertex_count, p_triangle_count,
            num_input_triangles);
    }

    void launch_orthogonalization(raw::vertex *vertices, size_t num_vertices,
                                  raw::cuda_types::cuda_stream &stream) {
        constexpr auto threads_per_block = 256;
        auto blocks = (num_vertices + threads_per_block - 1) / 256;
        orthogonalize<<<blocks, threads_per_block, 0, stream.stream()>>>(vertices, num_vertices);
    }
} // namespace raw::sphere_generation