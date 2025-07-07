//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include "mesh_generator.h"
#include <array>
#include <glm/gtc/matrix_transform.hpp>
namespace raw {
inline constexpr float				GOLDEN_RATIO = 1.61803398874989484820;
constexpr std::array<glm::vec3, 12> generate_icosahedron(float radius) {
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

			glm::vec3 point;
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
__global__ void loop_subdivision()
} // namespace raw
#endif // SPACE_EXPLORER_MESH_GENERATOR_H
