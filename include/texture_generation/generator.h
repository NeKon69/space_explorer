//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_GENERATOR_H
#define SPACE_EXPLORER_GENERATOR_H
#include <glm/glm.hpp>

#include "helper_macros.h"

namespace raw {

namespace texture {

struct data {
	// VBO's of the objects
	raw::UI albedo;
	raw::UI normal;
	raw::UI metallic;
	raw::UI roughness;
	raw::UI ao;
};

namespace size {
static constexpr auto MAX = glm::uvec2(4096, 2048);
static constexpr auto MID = glm::uvec2(2048, 1024);
static constexpr auto LOW = glm::uvec2(1024, 512);
static constexpr auto MIN = glm::uvec2(256, 128);
} // namespace size

class generator {
private:
raw::shared_ptr < public : generator() = default;
	void generate(size_t seed, raw::texture::data data, glm::uvec2 texture_size = size::MAX);
};
} // namespace texture
} // namespace raw

#endif // SPACE_EXPLORER_GENERATOR_H
