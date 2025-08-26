//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define SPACE_EXPLORER_TESSELLATION_KERNEL_H
#define GLM_CUDA_FORCE_DEVICE_FUNC

#include <glm/glm.hpp>

namespace raw::sphere_generation {
extern __global__ void subdivide(const raw::vertex *in_vertices, const unsigned int *in_indices,
								 raw::vertex *out_vertices, unsigned int *out_indices,
								 uint32_t *p_vertex_count, uint32_t *p_triangle_count,
								 size_t num_input_triangles);

extern __global__ void orthogonalize(raw::vertex *vertices, uint32_t vertex_count);
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_TESSELLATION_KERNEL_H