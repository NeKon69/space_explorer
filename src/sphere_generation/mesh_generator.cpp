//
// Created by progamers on 7/7/25.
//

#include "sphere_generation/mesh_generator.h"

#include <array>
#include <numbers>

#include "clock.h"
#include "cuda_types/buffer.h"
#include "sphere_generation/kernel_launcher.h"
#include "sphere_generation/tessellation_kernel.h"
namespace raw {
inline constexpr float GOLDEN_RATIO = std::numbers::phi_v<float>;
icosahedron_generator::icosahedron_generator(raw::UI vbo, raw::UI ebo, raw::UI steps,
											 float radius) {
	generate(vbo, ebo, steps, radius);
}

void icosahedron_generator::generate(raw::UI vbo, raw::UI ebo, raw::UI steps, float radius) {
	// FIXME: divide this block of code into two separate parts so i can make this thing actually
	// support LOD.
	// Mother fucker doesn't like then i use same amount of memory as was allocated so here will be
	// >= and not just >
	if (steps >= predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	size_t vertices_bytes = 0;
	vertices_handle		  = raw::make_shared<cuda_from_gl_data<glm::vec3>>(&vertices_bytes, vbo);

	size_t indices_bytes = 0;
	indices_handle		 = raw::make_shared<cuda_from_gl_data<UI>>(&indices_bytes, ebo);

	// FIXME: add those to the class members so there would be an ability to allocate/free them as
	// needed
	cuda_buffer<glm::vec3> vertices_second(vertices_bytes);
	cuda_buffer<UI>		   indices_second(indices_bytes);
	cuda_buffer<uint32_t>  amount_of_triangles(sizeof(uint32_t));
	cuda_buffer<uint32_t>  amount_of_vertices(sizeof(uint32_t));
	uint32_t			   num_vertices_cpu	 = 12;
	uint32_t			   num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;

	cudaMemcpy(vertices_handle->get_data(), (void*)std::data(generate_icosahedron_vertices(radius)),
			   num_triangles_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_handle->get_data(), (void*)std::data(generate_icosahedron_indices()),
			   num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice);
	raw::clock timer;
	for (UI i = 0; i < steps; ++i) {
		// FIXME: add some class functions to stop using those nasty ass cudaMemset funcs
		cudaMemset(amount_of_triangles.get(), 0, sizeof(uint32_t));
		cudaMemcpy(amount_of_vertices.get(), &num_vertices_cpu, sizeof(uint32_t),
				   cudaMemcpyHostToDevice);
		if (i % 2 == 0) {
			cudaMemcpy(vertices_second.get(), vertices_handle->get_data(),
					   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
			launch_tessellation(vertices_handle->get_data(), indices_handle->get_data(),
								vertices_second.get(), indices_second.get(),
								amount_of_vertices.get(), amount_of_triangles.get(),
								num_triangles_cpu, radius);
		} else {
			cudaMemcpy(vertices_handle->get_data(), vertices_second.get(),
					   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
			launch_tessellation(vertices_second.get(), indices_second.get(),
								vertices_handle->get_data(), indices_handle->get_data(),
								amount_of_vertices.get(), amount_of_triangles.get(),
								num_triangles_cpu, radius);
		}
		num_vertices_cpu += 3 * num_triangles_cpu;
		num_triangles_cpu *= 4;
		cudaDeviceSynchronize();
	}
	auto passed_time = timer.reset();
	passed_time.to_milli();
	std::cout << std::string("[Debug] Tesselation with amount of steps of ") << steps << " took "
			  << passed_time << " to complete\n";

	if (steps % 2 != 0) {
		cudaMemcpy(vertices_handle->get_data(), vertices_second.get(),
				   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
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
	return {2, 10, 4,  2, 4,  0, 2, 0, 5, 2, 5,	 11, 2, 11, 10, 0, 4, 8, 4, 10,
			6, 10, 11, 3, 11, 5, 7, 5, 0, 9, 1,	 8,	 6, 1,	6,	3, 1, 3, 7, 1,
			7, 9,  1,  9, 8,  6, 8, 4, 3, 6, 10, 7,	 3, 11, 9,	7, 5, 8, 9, 0};
}

constexpr std::pair<std::array<glm::vec3, 12>, std::array<UI, 60>>
icosahedron_generator::generate_icosahedron_data(float radius) {
	return std::pair {generate_icosahedron_vertices(radius), generate_icosahedron_indices()};
}

} // namespace raw
