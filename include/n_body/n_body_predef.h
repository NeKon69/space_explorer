//
// Created by progamers on 7/20/25.
//

#pragma once
#include "n_body/fwd.h"
namespace raw::n_body::predef {
static constexpr auto G = 0.001;

inline constexpr std::vector<space_object_data<float>> generate_data_for_sim() {
	std::vector<space_object_data<float>> data;
	data.reserve(5);

	data.emplace_back(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 1000.0f, 50.0f);

	data.emplace_back(glm::vec3(200.0f, 0.0f, 0.0f), glm::vec3(0.0f, 15.0f, 0.0f), 10.0f, 5.0f);

	data.emplace_back(glm::vec3(0.0f, -400.0f, 0.0f), glm::vec3(10.0f, 0.0f, 0.0f), 15.0f, 8.0f);

	data.emplace_back();
	data.back().position = glm::vec3(-100.0f, -100.0f, 0.0f);

	data.emplace_back(glm::vec3(210.0f, 10.0f, 0.0f), glm::vec3(0.0f, 18.0f, 0.0f), 0.5f, 1.0f);

	for (uint32_t i = 0; i < data.size(); ++i) {
		data[i].id = i;
	}

	return data;
}
} // namespace raw::n_body::predef
