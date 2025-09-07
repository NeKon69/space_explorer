//
// Created by progamers on 7/18/25.
//
#include <thrust/sort.h>

#include "graphics/vertex.h"
#include "sphere_generation/cuda/fwd.h"
#include "sphere_generation/cuda/tessellation_kernel.h"
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif
namespace raw::sphere_generation::cuda {
__device__ void make_canonical_edge(edge &edge, uint32_t i0, uint32_t i1) {
	edge.v0 = max(i0, i1);
	edge.v1 = min(i0, i1);
}
__global__ void generate_edges(const UI *in_indices, edge *out_edges, size_t num_input_triangles) {
	const UI x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= num_input_triangles) {
		return;
	}

	// Triangle's indices
	uint32_t i0 = in_indices[x * 3 + 0];
	uint32_t i1 = in_indices[x * 3 + 1];
	uint32_t i2 = in_indices[x * 3 + 2];

	edge *edge_ptr = &out_edges[x * 3];
	// Prepares edges for sorting
	make_canonical_edge(edge_ptr[0], i0, i1);
	make_canonical_edge(edge_ptr[1], i1, i2);
	make_canonical_edge(edge_ptr[2], i2, i0);
}

__global__ void create_unique_midpoint_vertices(
	const edge *sorted_edges, const graphics::vertex *in_vertices, graphics::vertex *out_vertices,
	uint32_t *p_vertex_count, edge *unique_edges, uint32_t *edge_to_vertex,
	uint32_t *num_unique_edges, size_t num_total_edges) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= num_total_edges) {
		return;
	}

	edge local_edge		 = sorted_edges[x];
	bool is_first_unique = false;

	// Since now we have sorted array first vertex will always be unique because it doesn't have
	// anything before it
	if (x == 0) {
		is_first_unique = true;
	} else {
		edge prev_edge = sorted_edges[x - 1];
		if (prev_edge != local_edge) {
			is_first_unique = true;
		}
	}
	if (is_first_unique) {
		// Get free spot in "unique_edges" array
		uint32_t unique_edge_id		 = atomicAdd(num_unique_edges, 1);
		unique_edges[unique_edge_id] = local_edge;

		uint32_t num_vertex_id = atomicAdd(p_vertex_count, 1);

		// Bind the spot to the vertex
		edge_to_vertex[unique_edge_id] = num_vertex_id;

		// Make usual calculations of the position/normal
		const graphics::vertex &v0 = in_vertices[local_edge.v0];
		const graphics::vertex &v1 = in_vertices[local_edge.v1];

		out_vertices[num_vertex_id].normal = out_vertices[num_vertex_id].position =
			glm::normalize(v0.position + v1.position);
	}
}

__global__ void sort_by_key(edge *d_unique_edges, const uint32_t *p_unique_edges_count,
							uint32_t *edge_to_vertex) {
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) {
		thrust::sort_by_key(thrust::device, d_unique_edges, d_unique_edges + *p_unique_edges_count,
							edge_to_vertex);
	}
}

// Uses binary sorting because "unique_edges" are sorted
__device__ uint32_t find_edge(const edge *unique_edges, uint32_t num_unique_edges, edge target) {
	uint32_t low  = 0;
	uint32_t high = num_unique_edges - 1;
	while (low <= high) {
		uint32_t mid	  = low + (high - low) / 2;
		edge	 mid_edge = unique_edges[mid];
		if (mid_edge.v0 < target.v0 || (mid_edge.v0 == target.v0 && mid_edge.v1 < target.v1)) {
			low = mid + 1;
		} else if (mid_edge == target) {
			return mid;
		} else {
			high = mid - 1;
		}
	}
	// we are so fucked
	return -1;
}

__global__ void create_triangles(const UI *in_indices, UI *out_indices, const edge *unique_edges,
								 const uint32_t *edge_to_vertex, const uint32_t *p_num_unique_edges,
								 const size_t num_input_triangles) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= num_input_triangles) {
		return;
	}
	const uint32_t num_unique_edges = *p_num_unique_edges;

	// retrieve existing indices
	const uint32_t i0 = in_indices[x * 3 + 0];
	const uint32_t i1 = in_indices[x * 3 + 1];
	const uint32_t i2 = in_indices[x * 3 + 2];

	// recreate the structure of the edges
	edge e01;
	edge e12;
	edge e20;
	make_canonical_edge(e01, i0, i1);
	make_canonical_edge(e12, i1, i2);
	make_canonical_edge(e20, i2, i0);

	// find the edge in "unique_edges" array
	int unique_id_01 = find_edge(unique_edges, num_unique_edges, e01);
	int unique_id_12 = find_edge(unique_edges, num_unique_edges, e12);
	int unique_id_20 = find_edge(unique_edges, num_unique_edges, e20);

	// retrieve new indices from the lookup
	uint32_t new_i01 = edge_to_vertex[unique_id_01];
	uint32_t new_i12 = edge_to_vertex[unique_id_12];
	uint32_t new_i20 = edge_to_vertex[unique_id_20];

	UI *out_tri_ptr = &out_indices[x * 12];
	// create 4 triangles (just imagine this in the brain, it gets pretty obvious)
	// first one is 0 - 01 - 20
	out_tri_ptr[0] = i0;
	out_tri_ptr[1] = new_i01;
	out_tri_ptr[2] = new_i20;
	// second one is 1 - 12 - 01
	out_tri_ptr[3] = i1;
	out_tri_ptr[4] = new_i12;
	out_tri_ptr[5] = new_i01;
	// third one is 2 - 20 - 12
	out_tri_ptr[6] = i2;
	out_tri_ptr[7] = new_i12;
	out_tri_ptr[8] = new_i20;
	// and fourth is 01 - 12 - 20
	out_tri_ptr[9]	= new_i01;
	out_tri_ptr[10] = new_i12;
	out_tri_ptr[11] = new_i20;
}
__global__ void calculate_tbn_and_uv(raw::graphics::vertex *vertices,
									 uint32_t			   *num_input_vertices) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *num_input_vertices) {
		return;
	}
	graphics::vertex &vertex = vertices[x];

	glm::vec3 P = vertex.position;

	float longitude = atan2f(P.z, P.x);
	float latitude	= asinf(P.y);

	vertex.tex_coord.x = longitude / (2.0f * CUDART_PI_F) + 0.5f;
	vertex.tex_coord.y = latitude / CUDART_PI_F + 0.5f;

	glm::vec3 up(0.0f, 1.0f, 0.0f);

	if (abs(P.y) > 0.9999f) {
		glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);
		vertex.tangent	= glm::normalize(glm::cross(vertex.normal, right));
	} else {
		vertex.tangent = glm::normalize(glm::cross(up, vertex.normal));
	}
	if (glm::dot(glm::cross(vertex.normal, vertex.tangent), vertex.bitangent) < 0.0f) {
		vertex.tangent = vertex.tangent * -1.0f;
	}
	vertex.bitangent = glm::normalize(glm::cross(vertex.normal, vertex.tangent));
}
/**
 *  @deprecated
 */
__device__ void calc_tex_coords(glm::vec2 *writing_ptr, const glm::vec3 &normalized_pos) {
	float u		 = atan2f(normalized_pos.z, normalized_pos.x) / (2.0f * CUDART_PI_F) + 0.5f;
	float v		 = 0.5f - asinf(normalized_pos.y) / CUDART_PI_F;
	*writing_ptr = glm::vec2 {u, v};
}
/**
 *  @deprecated
 */
__global__ void subdivide(raw::graphics::vertex *in_vertices, unsigned int *in_indices,
						  raw::graphics::vertex *out_vertices, unsigned int *out_indices,
						  uint32_t *p_vertex_count, uint32_t *p_triangle_count,
						  size_t num_input_triangles) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= num_input_triangles) {
		return;
	}

	size_t i0 = in_indices[x * 3 + 0];
	size_t i1 = in_indices[x * 3 + 1];
	size_t i2 = in_indices[x * 3 + 2];

	const raw::graphics::vertex &v0 = in_vertices[i0];
	const raw::graphics::vertex &v1 = in_vertices[i1];
	const raw::graphics::vertex &v2 = in_vertices[i2];

	glm::vec3 m01 = normalize(v0.position + v1.position);
	glm::vec3 m12 = normalize(v1.position + v2.position);
	glm::vec3 m20 = glm::normalize(v2.position + v0.position);

	uint32_t base_v_idx = atomicAdd(p_vertex_count, 3);
	uint32_t new_i01	= base_v_idx + 0;
	uint32_t new_i12	= base_v_idx + 1;
	uint32_t new_i20	= base_v_idx + 2;

	out_vertices[new_i01].position = normalize(m01);
	calc_tex_coords(&out_vertices[new_i01].tex_coord, out_vertices[new_i01].position);
	out_vertices[new_i12].position = normalize(m12);
	calc_tex_coords(&out_vertices[new_i12].tex_coord, out_vertices[new_i12].position);
	out_vertices[new_i20].position = normalize(m20);
	calc_tex_coords(&out_vertices[new_i20].tex_coord, out_vertices[new_i20].position);

	uint32_t	  base_t_idx  = atomicAdd(p_triangle_count, 4);
	unsigned int *out_tri_ptr = &out_indices[base_t_idx * 3];

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

	// Calculate tangent and bitangent for new triangles
	raw::graphics::vertex *new_triangle_vertices[4][3] = {
		{(raw::graphics::vertex *)&v0, &out_vertices[new_i01], &out_vertices[new_i20]},
		{(raw::graphics::vertex *)&v1, &out_vertices[new_i12], &out_vertices[new_i01]},
		{(raw::graphics::vertex *)&v2, &out_vertices[new_i20], &out_vertices[new_i12]},
		{&out_vertices[new_i01], &out_vertices[new_i12], &out_vertices[new_i20]}};
	for (const auto &new_triangle_vertice : new_triangle_vertices) {
		raw::graphics::vertex *v_a = new_triangle_vertice[0];
		raw::graphics::vertex *v_b = new_triangle_vertice[1];
		raw::graphics::vertex *v_c = new_triangle_vertice[2];

		glm::vec3 edge1		= v_b->position - v_a->position;
		glm::vec3 edge2		= v_c->position - v_a->position;
		glm::vec2 delta_uv1 = v_b->tex_coord - v_a->tex_coord;
		glm::vec2 delta_uv2 = v_c->tex_coord - v_a->tex_coord;

		// Some magic
		float f = 1.0f / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);

		glm::vec3 tangent, bitangent;

		// This is basically what i have to write
		//		tangent.x = f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x);
		//		tangent.y = f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y);
		//		tangent.z = f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z);

		// However this look much cooler and nicer
		for (int dim = 0; dim < 3; ++dim) {
			tangent[dim]   = f * (delta_uv2.y * edge1[dim] - delta_uv1.y * edge2[dim]);
			bitangent[dim] = f * (-delta_uv2.x * edge1[dim] + delta_uv1.x * edge2[dim]);
		}
		// This basically updates the values in the output buffers, but since we have places where
		// different threads work with same vertice, we need to use that ugly shit
		atomicAdd(&v_a->tangent.x, tangent.x);
		atomicAdd(&v_a->tangent.y, tangent.y);
		atomicAdd(&v_a->tangent.z, tangent.z);
		atomicAdd(&v_a->bitangent.x, bitangent.x);
		atomicAdd(&v_a->bitangent.y, bitangent.y);
		atomicAdd(&v_a->bitangent.z, bitangent.z);

		atomicAdd(&v_b->tangent.x, tangent.x);
		atomicAdd(&v_b->tangent.y, tangent.y);
		atomicAdd(&v_b->tangent.z, tangent.z);
		atomicAdd(&v_b->bitangent.x, bitangent.x);
		atomicAdd(&v_b->bitangent.y, bitangent.y);
		atomicAdd(&v_b->bitangent.z, bitangent.z);

		atomicAdd(&v_c->tangent.x, tangent.x);
		atomicAdd(&v_c->tangent.y, tangent.y);
		atomicAdd(&v_c->tangent.z, tangent.z);
		atomicAdd(&v_c->bitangent.x, bitangent.x);
		atomicAdd(&v_c->bitangent.y, bitangent.y);
		atomicAdd(&v_c->bitangent.z, bitangent.z);
	}
}
/**
 * @deprecated
 */
__global__ void orthogonalize(raw::graphics::vertex *vertices, uint32_t vertex_count) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x > vertex_count) {
		return;
	}
	raw::graphics::vertex &v	   = vertices[x];
	glm::vec3			   tangent = v.tangent - v.normal * dot(v.tangent, v.normal);

	// Basically zero
	if (length(tangent) < 1e-6) {
		if (abs(v.normal.x) > abs(v.normal.z)) {
			tangent = glm::vec3(-v.normal.y, v.normal.x, 0.0f);
		} else {
			tangent = glm::vec3(0.0f, -v.normal.z, v.normal.y);
		}
	}
	v.tangent = normalize(tangent);

	if (dot(cross(v.normal, v.tangent), v.bitangent) < 0.0f) {
		v.tangent = v.tangent * -1.0f;
	}
	v.bitangent = normalize(cross(v.normal, v.tangent));

	// Just for safe measure
	v.normal = normalize(v.normal);
}
} // namespace raw::sphere_generation::cuda