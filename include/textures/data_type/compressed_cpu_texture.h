//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_COMPRESSED_CPU_TEXTURE_H
#define SPACE_EXPLORER_COMPRESSED_CPU_TEXTURE_H
#include <vector>

#include "textures/fwd.h"

namespace raw::textures::data_type {
struct compressed_cpu_texture {
	std::vector<unsigned char> texture_data;
};
} // namespace raw::textures::data_type
#endif // SPACE_EXPLORER_COMPRESSED_CPU_TEXTURE_H