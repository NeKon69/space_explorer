//
// Created by progamers on 7/7/25.
//

#include "sphere_generation/mesh_generator.h"

#include <numbers>

#include "core/clock.h"
#include "cuda_types/buffer.h"
#include "sphere_generation/kernel_launcher.h"

namespace raw::sphere_generation {
inline constexpr float GOLDEN_RATIO = std::numbers::phi_v<float>;

icosahedron_data_manager::icosahedron_data_manager()
	: stream(make_shared<cuda_types::cuda_stream>()),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true),
	  amount_of_edges(sizeof(uint32_t), stream, true) {}

icosahedron_data_manager::icosahedron_data_manager(raw::UI vbo, raw::UI ebo, raw::UI steps)
	: stream(make_shared<cuda_types::cuda_stream>()),
	  amount_of_triangles(sizeof(uint32_t), stream, true),
	  amount_of_vertices(sizeof(uint32_t), stream, true),
	  amount_of_edges(sizeof(uint32_t), stream, true) {
	_vbo = vbo;
	_ebo = ebo;
	generate(vbo, ebo, steps);
}

void icosahedron_data_manager::init(raw::UI vbo, raw::UI ebo) {
	_vbo			= vbo;
	_ebo			= ebo;
	vertices_handle = cuda_types::cuda_from_gl_data<raw::graphics::vertex>(&vertices_bytes, vbo);
	indices_handle	= cuda_types::cuda_from_gl_data<UI>(&indices_bytes, ebo);

	vertices_second = cuda_types::cuda_buffer<raw::graphics::vertex>(vertices_bytes, stream, true);
	indices_second	= cuda_types::cuda_buffer<UI>(indices_bytes, stream, true);
	all_edges = cuda_types::cuda_buffer<edge>(predef::MAXIMUM_AMOUNT_OF_TRIANGLES * sizeof(edge),
											  stream, true);
	edge_to_vertex = cuda_types::cuda_buffer<uint32_t>(
		predef::MAXIMUM_AMOUNT_OF_TRIANGLES * sizeof(uint32_t), stream, true);
	d_unique_edges = cuda_types::cuda_buffer<edge>(
		predef::MAXIMUM_AMOUNT_OF_TRIANGLES * sizeof(edge), stream, true);
	amount_of_edges.zero_data(sizeof(uint32_t));

	inited = true;
}

void icosahedron_data_manager::prepare(raw::UI vbo, raw::UI ebo) {
	if (!inited) {
		init(vbo, ebo);
		return;
	}
	vertices_handle.map();
	indices_handle.map();
	vertices_second.allocate(vertices_bytes);
	indices_second.allocate(indices_bytes);
	// Need to update func to also produce some cool data as tangent/bitangent
	cudaMemcpy(vertices_handle.get_data(), (void *)std::data(generate_icosahedron_vertices()),
			   num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(indices_handle.get_data(), (void *)std::data(generate_icosahedron_indices()),
			   num_triangles_cpu * 3 * sizeof(UI), cudaMemcpyHostToDevice);
}
// TODO: Delete this function as this class will serve only as data manager
void icosahedron_data_manager::generate(raw::UI vbo, raw::UI ebo, raw::UI steps) {
	// Motherfucker doesn't like then i use same amount of memory as was allocated so here will be
	// >= and not just > (which sucks btw)
	// One day i will find that sick piece of shit that doesn't let me use all allocated memory, and
	// i promise, if he even then will say cudaErrorIllegalAddress, i will kill myself
	// TODO: Put this check into generator function
	if (steps >= predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	// TODO: After refactoring the class remember to delete that
	if (vbo != _vbo || ebo != _ebo) {
		throw std::runtime_error(std::format(
			"Function for LOD on different BO's was not yet created, don't call that. VBO given was {} while stored VBO was {}, EBO given was {} while stored was {}",
			vbo, _vbo, ebo, _ebo));
	}

	prepare(vbo, ebo);

	raw::core::clock timer;
	for (UI i = 0; i < steps; ++i) {
		amount_of_triangles.zero_data(sizeof(UI) * 1);
		amount_of_vertices.set_data(&num_vertices_cpu, sizeof(uint32_t), cudaMemcpyHostToDevice);
		if (i % 2 == 0) {
			vertices_second.set_data(vertices_handle.get_data(),
									 num_vertices_cpu * sizeof(glm::vec3),
									 cudaMemcpyDeviceToDevice);
			launch_tessellation(vertices_handle.get_data(), indices_handle.get_data(),
								vertices_second.get(), indices_second.get(),
								amount_of_vertices.get(), amount_of_triangles.get(),
								num_triangles_cpu, *stream);
		} else {
			vertices_second.set_data(vertices_handle.get_data(),
									 num_vertices_cpu * sizeof(glm::vec3), cudaMemcpyDeviceToDevice,
									 cuda_types::cudaMemcpyOrder::cudaMemcpy1to2);
			launch_tessellation(vertices_second.get(), indices_second.get(),
								vertices_handle.get_data(), indices_handle.get_data(),
								amount_of_vertices.get(), amount_of_triangles.get(),
								num_triangles_cpu, *stream);
		}
		num_vertices_cpu += 3 * num_triangles_cpu;
		num_triangles_cpu *= 4;
	}
	// eat my ass (for no reason i am just sleepy)
	if (steps % 2 != 0) {
		vertices_second.set_data(vertices_handle.get_data(), num_vertices_cpu * sizeof(glm::vec3),
								 cudaMemcpyDeviceToDevice,
								 cuda_types::cudaMemcpyOrder::cudaMemcpy1to2);
	}
	launch_orthogonalization(vertices_handle.get_data(), num_vertices_cpu, *stream);
	stream->sync();
	auto passed_time = timer.restart();
	passed_time.to_milli();
	std::cout << std::string("[Debug] Tesselation with amount of steps of ") << steps << " took "
			  << passed_time << " to complete\n";

	cleanup();
}

void icosahedron_data_manager::cleanup() {
	stream->sync();
	vertices_second.free();
	indices_second.free();
	vertices_handle.unmap();
	indices_handle.unmap();
	num_vertices_cpu  = 12;
	num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;
}

// Icosahedron as most things in this project sounds horrifying, but, n-body (my algorithms), this
// thing, and tesselation are surprisingly easy things, just for some reason someone wanted give
// them scary names
constexpr std::array<glm::vec3, 12> icosahedron_data_manager::generate_icosahedron_vertices() {
	std::array<glm::vec3, 12> vertices {};
	int						  vertex_index = 0;

	// FIXME: program won't launch properly until i actually 100% not clickbait will actually
	// eventually fix this to return not only the vertices itself but also other important stuff (i
	// also hate opengl and want to make my own renderer)
	const float unscaled_dist = std::sqrt(1.0f + GOLDEN_RATIO * GOLDEN_RATIO);
	const float scale		  = 1 / unscaled_dist;
	const float a			  = 1.0f * scale;
	const float b			  = GOLDEN_RATIO * scale;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			const auto sign1 = (j & 2) ? -1.0f : 1.0f;
			const auto sign2 = (j & 1) ? -1.0f : 1.0f;

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

constexpr std::array<UI, 60> icosahedron_data_manager::generate_icosahedron_indices() {
	return {2, 10, 4,  2, 4,  0, 2, 0, 5, 2, 5,	 11, 2, 11, 10, 0, 4, 8, 4, 10,
			6, 10, 11, 3, 11, 5, 7, 5, 0, 9, 1,	 8,	 6, 1,	6,	3, 1, 3, 7, 1,
			7, 9,  1,  9, 8,  6, 8, 4, 3, 6, 10, 7,	 3, 11, 9,	7, 5, 8, 9, 0};
}

constexpr std::pair<std::array<glm::vec3, 12>, std::array<UI, 60> >
icosahedron_data_manager::generate_icosahedron_data() {
	return std::pair {generate_icosahedron_vertices(), generate_icosahedron_indices()};
}
} // namespace raw::sphere_generation