//
// Created by progamers on 7/20/25.
//

#pragma once
#include <vector>

#include "n_body/fwd.h"
namespace raw::n_body::predef {
static constexpr auto G = 100;

inline constexpr std::vector<space_object_data<float>> generate_data_for_sim() {
	std::vector<space_object_data<float>> data;
	data.reserve(3);

	data.emplace_back(glm::vec3(5.0));

	data.emplace_back(glm::vec3(-5.0));

	data.emplace_back(glm::vec3(7.5, 0.0f, 0.0f));

	for (uint32_t i = 0; i < data.size(); ++i) {
		data[i].id = i;
	}

	return data;
}
} // namespace raw::n_body::predef
