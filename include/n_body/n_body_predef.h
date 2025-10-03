//
// Created by progamers on 7/20/25.
//

#pragma once
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include <random>
#include <vector>

#include "n_body/fwd.h"
#include "n_body/physics_component.h"

namespace raw::n_body::predef {
static constexpr auto G = 0.1;

inline decltype(auto) generate_data_for_sim() {
	std::vector<space_object_data<float>> data;
	std::vector<physics_component>		  components;

	const int	resolution = 7;
	const float spacing	   = 8.0f;
	const float mass	   = 5.0f;
	data.reserve(resolution * resolution * resolution);
	components.resize(resolution * resolution * resolution);

	const float cube_half_size = (resolution - 1) * spacing / 2.0f;

	for (int x = 0; x < resolution; ++x) {
		for (int y = 0; y < resolution; ++y) {
			for (int z = 0; z < resolution; ++z) {
				glm::vec3 pos(x * spacing, y * spacing, z * spacing);
				pos -= glm::vec3(cube_half_size);

				data.emplace_back(pos, glm::vec3(0.0f), mass, 0.5f);
			}
		}
	}
	return std::tuple {data, components};
}

} // namespace raw::n_body::predef
