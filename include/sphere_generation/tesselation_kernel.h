//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_TESSELATION_KERNEL_H
#define SPACE_EXPLORER_TESSELATION_KERNEL_H
#define GLM_CUDA_FORCE_DEVICE_FUNC

#include <glm/glm.hpp>

extern __global__ void subdivide(const glm::vec3* in_vertices, const unsigned int* in_indices,
								 glm::vec3* out_vertices, unsigned int* out_indices,
								 uint32_t* p_vertex_count, uint32_t* p_triangle_count,
								 size_t num_input_triangles, float radius);

#endif // SPACE_EXPLORER_TESSELATION_KERNEL_H
