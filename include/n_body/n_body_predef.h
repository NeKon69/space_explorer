//
// Created by progamers on 7/20/25.
//

#pragma once
#include <vector>

#include "n_body/fwd.h"
#include "n_body/physics_component.h"
namespace raw::n_body::predef {
static constexpr auto G = 100;

inline constexpr std::vector<physics_component<float>> generate_data_for_sim() {
	std::vector<physics_component<float>> data;
	data.reserve(3);

	data.emplace_back(glm::vec3(5.0));

	data.emplace_back(glm::vec3(-5.0));

	data.emplace_back(glm::vec3(7.5, 0.0f, 0.0f));

	return data;
}
} // namespace raw::n_body::predef
