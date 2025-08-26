//
// Created by progamers on 7/18/25.
//
#include "../../include/graphics/vertex.h"
#include "sphere_generation/tessellation_kernel.h"
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif
__device__ void calc_tex_coords(glm::vec2 *writing_ptr, glm::vec3 &normalized_pos) {
    float u = atan2f(normalized_pos.z, normalized_pos.x) / (2.0f * CUDART_PI_F) + 0.5f;
    float v = 0.5f - asinf(normalized_pos.y) / CUDART_PI_F;
    *writing_ptr = glm::vec2{u, v};
}

__global__ void subdivide(const raw::vertex *in_vertices, const unsigned int *in_indices,
                          raw::vertex *out_vertices, unsigned int *out_indices,
                          uint32_t *p_vertex_count, uint32_t *p_triangle_count,
                          size_t num_input_triangles) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= num_input_triangles) {
        return;
    }

    size_t i0 = in_indices[x * 3 + 0];
    size_t i1 = in_indices[x * 3 + 1];
    size_t i2 = in_indices[x * 3 + 2];

    const raw::vertex &v0 = in_vertices[i0];
    const raw::vertex &v1 = in_vertices[i1];
    const raw::vertex &v2 = in_vertices[i2];

    glm::vec3 m01 = normalize(v0.position + v1.position);
    glm::vec3 m12 = normalize(v1.position + v2.position);
    glm::vec3 m20 = normalize(v2.position + v0.position);

    uint32_t base_v_idx = atomicAdd(p_vertex_count, 3);
    uint32_t new_i01 = base_v_idx + 0;
    uint32_t new_i12 = base_v_idx + 1;
    uint32_t new_i20 = base_v_idx + 2;

    out_vertices[new_i01].position = normalize(m01);
    out_vertices[new_i01].tex_coord = normalize(m01);
    calc_tex_coords(&out_vertices[new_i01].tex_coord, out_vertices[new_i01].position);
    out_vertices[new_i12].position = normalize(m12);
    out_vertices[new_i01].tex_coord = normalize(m12);
    calc_tex_coords(&out_vertices[new_i12].tex_coord, out_vertices[new_i12].position);
    out_vertices[new_i20].position = normalize(m20);
    out_vertices[new_i01].tex_coord = normalize(m20);
    calc_tex_coords(&out_vertices[new_i20].tex_coord, out_vertices[new_i20].position);

    uint32_t base_t_idx = atomicAdd(p_triangle_count, 4);
    unsigned int *out_tri_ptr = &out_indices[base_t_idx * 3];

    out_tri_ptr[0] = i0;
    out_tri_ptr[1] = new_i01;
    out_tri_ptr[2] = new_i20;
    out_tri_ptr[3] = i1;
    out_tri_ptr[4] = new_i12;
    out_tri_ptr[5] = new_i01;
    out_tri_ptr[6] = i2;
    out_tri_ptr[7] = new_i20;
    out_tri_ptr[8] = new_i12;
    out_tri_ptr[9] = new_i01;
    out_tri_ptr[10] = new_i12;
    out_tri_ptr[11] = new_i20;

    // Calculate tangent and bitangent for new triangles
    raw::vertex *new_triangle_vertices[4][3] = {
        {(raw::vertex *) &v0, &out_vertices[new_i01], &out_vertices[new_i20]},
        {(raw::vertex *) &v1, &out_vertices[new_i12], &out_vertices[new_i01]},
        {(raw::vertex *) &v2, &out_vertices[new_i20], &out_vertices[new_i12]},
        {&out_vertices[new_i01], &out_vertices[new_i12], &out_vertices[new_i20]}
    };
    for (const auto &new_triangle_vertice: new_triangle_vertices) {
        raw::vertex *v_a = new_triangle_vertice[0];
        raw::vertex *v_b = new_triangle_vertice[1];
        raw::vertex *v_c = new_triangle_vertice[2];

        glm::vec3 edge1 = v_b->position - v_a->position;
        glm::vec3 edge2 = v_c->position - v_a->position;
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
            tangent[dim] = f * (delta_uv2.y * edge1[dim] - delta_uv1.y * edge2[dim]);
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

__global__ void orthogonalize(raw::vertex *vertices, uint32_t vertex_count) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > vertex_count) {
        return;
    }
    raw::vertex &v = vertices[x];
    glm::vec3 tangent = v.tangent - v.normal * dot(v.tangent, v.normal);

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