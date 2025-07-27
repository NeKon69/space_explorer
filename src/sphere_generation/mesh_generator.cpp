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
icosahedron_generator::icosahedron_generator()
	: stream(make_shared<cuda_stream>()),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true) {}
icosahedron_generator::icosahedron_generator(raw::UI vbo, raw::UI tex_coord_vbo, raw::UI ebo,
											 raw::UI steps)
	: stream(make_shared<cuda_stream>()),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true) {
	_vbo		   = vbo;
	_tex_coord_vbo = tex_coord_vbo;
	_ebo		   = ebo;
	generate(vbo, tex_coord_vbo, ebo, steps);
}

void icosahedron_generator::init(raw::UI vbo, raw::UI tex_coord_vbo, raw::UI ebo) {
	_vbo			 = vbo;
	_tex_coord_vbo	 = tex_coord_vbo;
	_ebo			 = ebo;
	vertices_handle	 = cuda_from_gl_data<glm::vec3>(&vertices_bytes, vbo);
	tex_coord_handle = cuda_from_gl_data<glm::vec2>(&tex_coords_bytes, tex_coord_vbo);
	indices_handle	 = cuda_from_gl_data<UI>(&indices_bytes, ebo);

	vertices_second = cuda_buffer<glm::vec3>(vertices_bytes, stream, true);
	indices_second	= cuda_buffer<UI>(indices_bytes, stream, true);

	cudaMemcpy(vertices_handle.get_data(), (void*)std::data(generate_icosahedron_vertices()),
			   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_handle.get_data(), (void*)std::data(generate_icosahedron_indices()),
			   num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice);
	inited = true;
}

void icosahedron_generator::prepare(raw::UI vbo, raw::UI tex_coord_vbo, raw::UI ebo) {
	if (!inited) {
		init(vbo, tex_coord_vbo, ebo);
		return;
	}
	vertices_handle.map();
	tex_coord_handle.map();
	indices_handle.map();
	vertices_second.allocate(vertices_bytes);
	indices_second.allocate(indices_bytes);
	cudaMemcpy(vertices_handle.get_data(), (void*)std::data(generate_icosahedron_vertices()),
			   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_handle.get_data(), (void*)std::data(generate_icosahedron_indices()),
			   num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice);
}

void icosahedron_generator::generate(raw::UI vbo, raw::UI tex_coord_vbo, raw::UI ebo,
									 raw::UI steps) {
	// Motherfucker doesn't like then i use same amount of memory as was allocated so here will be
	// >= and not just > (which sucks btw)
	// One day i will find that sick piece of shit that doesn't let me use all allocated memory, and
	// i promise, if he even then will say cudaErrorIllegalAddress, i will kill myself
	if (steps >= predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	if (vbo != _vbo || ebo != _ebo || _tex_coord_vbo != tex_coord_vbo) {
		throw std::runtime_error(std::format(
			"Function for LOD on different BO's was not yet created, don't call that. VBO given was {} while stored VBO was {}, EBO given was {} while stored was {}, tex_coord_VBO was given {} while was stored {}",
			vbo, _vbo, ebo, _ebo, tex_coord_vbo, _tex_coord_vbo));
	}

	prepare(vbo, tex_coord_vbo, ebo);

	raw::clock timer;
	for (UI i = 0; i < steps; ++i) {
		amount_of_triangles.zero_data(sizeof(UI) * 1);
		amount_of_vertices.set_data(&num_vertices_cpu, sizeof(uint32_t), cudaMemcpyHostToDevice);
		if (i % 2 == 0) {
			vertices_second.set_data(vertices_handle.get_data(),
									 num_vertices_cpu * sizeof(glm::vec3),
									 cudaMemcpyDeviceToDevice);
			launch_tessellation(vertices_handle.get_data(), indices_handle.get_data(),
								vertices_second.get(), tex_coord_handle.get_data(),
								indices_second.get(), amount_of_vertices.get(),
								amount_of_triangles.get(), num_triangles_cpu, *stream);
		} else {
			vertices_second.set_data(vertices_handle.get_data(),
									 num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyDeviceToDevice,
									 cudaMemcpyOrder::cudaMemcpy1to2);
			launch_tessellation(vertices_second.get(), indices_second.get(),
								vertices_handle.get_data(), tex_coord_handle.get_data(),
								indices_handle.get_data(), amount_of_vertices.get(),
								amount_of_triangles.get(), num_triangles_cpu, *stream);
		}
		num_vertices_cpu += 3 * num_triangles_cpu;
		num_triangles_cpu *= 4;
	}
	stream->sync();
	auto passed_time = timer.restart();
	passed_time.to_milli();
	std::cout << std::string("[Debug] Tesselation with amount of steps of ") << steps << " took "
			  << passed_time << " to complete\n";

	if (steps % 2 != 0) {
		vertices_second.set_data(vertices_handle.get_data(), num_vertices_cpu * sizeof(glm::vec3),
								 cudaMemcpyDeviceToDevice, cudaMemcpyOrder::cudaMemcpy1to2);
	}
	cleanup();
}
void icosahedron_generator::cleanup() {
	stream->sync();
	vertices_second.free();
	indices_second.free();
	vertices_handle.unmap();
	tex_coord_handle.unmap();
	indices_handle.unmap();
	num_vertices_cpu   = 12;
	num_tex_coords_cpu = 12;
	num_triangles_cpu  = predef::BASIC_AMOUNT_OF_TRIANGLES;
}
// Icosahedron as most things in this project sounds horrifying, but, n-body (my algorithms), this
// thing, and tesselation are surprisingly easy things, just for some reason someone wanted give
// them scary names
constexpr std::array<glm::vec3, 12> icosahedron_generator::generate_icosahedron_vertices() {
	std::array<glm::vec3, 12> vertices {};
	int						  vertex_index = 0;

	const float unscaled_dist = std::sqrt(1.0f + GOLDEN_RATIO * GOLDEN_RATIO);
	const float scale		  = 1 / unscaled_dist;
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
icosahedron_generator::generate_icosahedron_data() {
	return std::pair {generate_icosahedron_vertices(), generate_icosahedron_indices()};
}

} // namespace raw
