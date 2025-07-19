//
// Created by progamers on 7/18/25.
//
#include <printf.h>

#include "sphere_generation/tesselation_kernel.h"

__global__ void subdivide(const glm::vec3* in_vertices, const unsigned int* in_indices,
						  glm::vec3* out_vertices, unsigned int* out_indices,
						  uint32_t* p_vertex_count, uint32_t* p_triangle_count,
						  size_t num_input_triangles, float radius) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= num_input_triangles) {
		return;
	}

	size_t i0 = in_indices[x * 3 + 0];
	size_t i1 = in_indices[x * 3 + 1];
	size_t i2 = in_indices[x * 3 + 2];

	glm::vec3 v0 = in_vertices[i0];
	glm::vec3 v1 = in_vertices[i1];
	glm::vec3 v2 = in_vertices[i2];

	glm::vec3 m01 = normalize(v0 + v1) * radius;
	glm::vec3 m12 = normalize(v1 + v2) * radius;
	glm::vec3 m20 = normalize(v2 + v0) * radius;

	uint32_t base_v_idx = atomicAdd(p_vertex_count, 3);
	uint32_t new_i01	= base_v_idx + 0;
	uint32_t new_i12	= base_v_idx + 1;
	uint32_t new_i20	= base_v_idx + 2;

	out_vertices[new_i01] = m01;
	out_vertices[new_i12] = m12;
	out_vertices[new_i20] = m20;

	uint32_t	  base_t_idx  = atomicAdd(p_triangle_count, 4);
	unsigned int* out_tri_ptr = &out_indices[base_t_idx * 3];

	out_tri_ptr[0]	= i0;
	out_tri_ptr[1]	= new_i01;
	out_tri_ptr[2]	= new_i20;
	out_tri_ptr[3]	= i1;
	out_tri_ptr[4]	= new_i12;
	out_tri_ptr[5]	= new_i01;
	out_tri_ptr[6]	= i2;
	out_tri_ptr[7]	= new_i20;
	out_tri_ptr[8]	= new_i12;
	out_tri_ptr[9]	= new_i01;
	out_tri_ptr[10] = new_i12;
	out_tri_ptr[11] = new_i20;

	__syncthreads();

	if (x == 1) {
		*p_triangle_count = 0;
		*p_vertex_count	  = 0;
	}
}