//
// Created by progamers on 9/5/25.
//

#ifndef SPACE_EXPLORER_BASIC_GEOMERTY_H
#define SPACE_EXPLORER_BASIC_GEOMERTY_H
#include "graphics/vertex.h"
#include "sphere_generation/fwd.h"
namespace raw::sphere_generation {
inline constexpr float GOLDEN_RATIO = std::numbers::phi_v<float>;
inline constexpr float PI			= std::numbers::pi_v<float>;
// Icosahedron as most things in this project sounds horrifying, but, n-body (my algorithms), this
// thing, and tesselation are surprisingly easy things, just for some reason someone wanted give
// them scary names
constexpr std::array<graphics::vertex, predef::BASIC_AMOUNT_OF_VERTICES>
generate_icosahedron_vertices() {
	std::array<graphics::vertex, predef::BASIC_AMOUNT_OF_VERTICES> vertices;
	int															   vertex_index = 0;

	const float unscaled_dist = std::sqrt(1.0f + GOLDEN_RATIO * GOLDEN_RATIO);
	const float scale		  = 1 / unscaled_dist;
	const float a			  = 1.0f * scale;
	const float b			  = GOLDEN_RATIO * scale;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			const auto sign1 = (j & 2) ? -1.0f : 1.0f;
			const auto sign2 = (j & 1) ? -1.0f : 1.0f;

			glm::vec3 point(1.0f);
			if (i == 0) {
				point = {sign1 * a, sign2 * b, 0.0f};
			} else if (i == 1) {
				point = {0.0f, sign1 * a, sign2 * b};
			} else {
				point = {sign1 * b, 0.0f, sign2 * a};
			}

			auto &v = vertices[vertex_index++];

			v.position = glm::normalize(point);
			v.normal   = v.position;

			constexpr glm::vec3 up = {0.0f, 1.0f, 0.0f};
			v.tangent			   = glm::normalize(glm::cross(up, v.normal));
			v.bitangent			   = glm::normalize(glm::cross(v.normal, v.tangent));
			v.tex_coord.x		   = 0.5f + std::atan2(v.normal.z, v.normal.x) / (2.0f * PI);
			v.tex_coord.y		   = 0.5f - std::asin(v.normal.y) / PI;
		}
	}
	return vertices;
}

constexpr std::array<UI, predef::BASIC_AMOUNT_OF_INDICES> generate_icosahedron_indices() {
	return {2, 10, 4,  2, 4,  0, 2, 0, 5, 2, 5,	 11, 2, 11, 10, 0, 4, 8, 4, 10,
			6, 10, 11, 3, 11, 5, 7, 5, 0, 9, 1,	 8,	 6, 1,	6,	3, 1, 3, 7, 1,
			7, 9,  1,  9, 8,  6, 8, 4, 3, 6, 10, 7,	 3, 11, 9,	7, 5, 8, 9, 0};
}
} // namespace raw::sphere_generation

#endif // SPACE_EXPLORER_BASIC_GEOMERTY_H
