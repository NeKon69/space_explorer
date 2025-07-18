//
// Created by progamers on 7/7/25.
//

#include "sphere_generation/mesh_generator.h"

#include <array>

#include "sphere_generation/cuda_buffer.h"
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tesselation_kernel.h"
namespace raw {
inline constexpr float GOLDEN_RATIO = 1.61803398874989484820;
icosahedron_generator::icosahedron_generator(raw::UI vbo, raw::UI ebo, raw::UI steps,
											 float radius) {
	generate(vbo, ebo, steps, radius);
}

void icosahedron_generator::generate(raw::UI vbo, raw::UI ebo, raw::UI steps, float radius) {
	// FIXME: divide this block of code into two separate parts so i can make this thing actually
	// support LOD Since data should end up in the final buffer, we put that here
	if (steps % 2 != 0) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should be multiple of two to launch properly, while was given: {}",
			steps));
	}
	if (steps > predef::MAX_STEPS) {
		throw std::runtime_error(std::format("[Error] Amount of steps should not exceed maximum, which is {}, while was given {}", predef::MAX_STEPS, steps));
	}
	glm::vec3* vertices = nullptr;
	size_t	   vertices_bytes;
	vertices_handle = raw::make_unique<cuda_from_gl_data>(vertices, &vertices_bytes, vbo);

	UI*	   indices = nullptr;
	size_t indices_bytes;
	indices_handle = raw::make_unique<cuda_from_gl_data>(indices, &indices_bytes, ebo);

	// FIXME: add those to the class members so there would be an ability to allocate/free them as
	// needed
	cuda_buffer<glm::vec3> vertices_second(vertices_bytes);
	cuda_buffer<UI>		   indices_second(indices_bytes);
	cuda_buffer<uint32_t>  amount_of_triangles(sizeof(uint32_t));
	cuda_buffer<uint32_t>  amount_of_vertices(sizeof(uint32_t));

	UI amount_of_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;

	assign_data(indices, vertices, radius);
	for (UI i = 0; i < steps; ++i) {
		if (i % 2 == 0) {
			launch_tesselation(vertices, indices, vertices_second.get(), indices_second.get(),
							   amount_of_vertices.get(), amount_of_triangles.get(),
							   amount_of_triangles_cpu, radius);
		} else {
			launch_tesselation(vertices_second.get(), indices_second.get(), vertices, indices,
							   amount_of_vertices.get(), amount_of_triangles.get(),
							   amount_of_triangles_cpu, radius);
		}

		amount_of_triangles_cpu = 20 * std::pow(4, i);
	}

	vertices_handle->unmap();
	indices_handle->unmap();
}

constexpr std::array<glm::vec3, 12> icosahedron_generator::generate_icosahedron_vertices(
	float radius) {
	std::array<glm::vec3, 12> vertices {};
	int						  vertex_index = 0;

	const float unscaled_dist = std::sqrt(1.0f + GOLDEN_RATIO * GOLDEN_RATIO);
	const float scale		  = radius / unscaled_dist;
	const float a			  = 1.0f * scale;
	const float b			  = GOLDEN_RATIO * scale;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			const float sign1 = (j & 2) ? -1.0f : 1.0f;
			const float sign2 = (j & 1) ? -1.0f : 1.0f;

			glm::vec3 point = {1.0f, 1.0f, 1.0f};
			if (i == 0) {
				point = {sign1 * a, sign2 * b, 0.0f};
			} else if (i == 1) {
				point = {0.0f, sign1 * a, sign2 * b};
			} else {
				point = {sign1 * b, 0.0f, sign2 * a};
			}

			vertices[vertex_index++] = point;
		}
	}
	return vertices;
}
constexpr std::array<UI, 60> icosahedron_generator::generate_icosahedron_indices() {
	return {0,	4, 1, 0, 9, 4, 9, 5,  4, 4, 5,	8,	4,	8, 1, 8,  10, 1,  8, 3,
			10, 5, 3, 8, 5, 2, 3, 2,  7, 3, 7,	10, 3,	7, 6, 10, 7,  11, 6, 11,
			0,	6, 0, 1, 6, 6, 1, 10, 9, 0, 11, 9,	11, 2, 9, 2,  5,  7,  2, 11};
}

constexpr std::pair<std::array<glm::vec3, 12>, std::array<UI, 60>>
icosahedron_generator::generate_icosahedron_data(float radius) {
	return std::pair {generate_icosahedron_vertices(radius), generate_icosahedron_indices()};
}

void icosahedron_generator::assign_data(raw::UI* indices, glm::vec3* vertices, float radius) {
	auto vertices_arr = generate_icosahedron_vertices(radius);
	auto indices_arr  = generate_icosahedron_indices();
	for (UI i = 0; i < vertices_arr.size(); ++i) {
		vertices[i] = vertices_arr[i];
	}
	for (UI i = 0; i < indices_arr.size(); ++i) {
		indices[i] = indices_arr[i];
	}
}
} // namespace raw
