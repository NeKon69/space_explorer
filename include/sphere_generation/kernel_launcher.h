//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_KERNEL_LAUNCHER_H
#define SPACE_EXPLORER_KERNEL_LAUNCHER_H
#include <glm/glm.hpp>

#include "cuda_types/stream.h"
#include "helper_macros.h"

namespace raw {
extern void launch_tessellation(const glm::vec3* in_vertices, const UI* in_indices,
								glm::vec3* out_vertices, UI* out_indices, uint32_t* p_vertex_count,
								uint32_t* p_triangle_count, size_t num_input_triangles,
								float radius, cuda_stream& stream);
}

#endif // SPACE_EXPLORER_KERNEL_LAUNCHER_H
