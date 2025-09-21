//
// Created by progamers on 7/20/25.
//

#pragma once
#include <vector>

#include "n_body/fwd.h"
#include "n_body/physics_component.h"
namespace raw::n_body::predef {
static constexpr auto G = 100;

inline constexpr decltype(auto) generate_data_for_sim() {
	std::vector<space_object_data<float>> data;
	std::vector<physics_component>		  components;
	data.reserve(3);

	data.emplace_back(glm::vec3(5.0));

	data.emplace_back(glm::vec3(-5.0));

	data.emplace_back(glm::vec3(7.5, 0.0f, 0.0f));

	return std::tuple {data, components};
}
} // namespace raw::n_body::predef
