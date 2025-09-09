//
// Created by progamers on 8/5/25.
//

#pragma once
#include <vector>

#include "textures/fwd.h"

namespace raw::textures::data_type {
struct compressed_cpu_texture {
	std::vector<unsigned char> texture_data;
};
} // namespace raw::textures::data_type
