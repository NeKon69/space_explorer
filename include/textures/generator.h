//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_GENERATOR_H
#define SPACE_EXPLORER_GENERATOR_H
#include <glm/glm.hpp>

#include "helper_macros.h"
#include "textures/fwd.h"
#include "textures/planet_dna.h"

namespace raw::texture {
// Acts as a host side launcher
// Also does some checks to not compile if expected "output_data" byte size doesn't match the byte
// size expected
void generate(raw::texture::data output_data, raw::texture_generation::planet_dna planet_data,
			  raw::UI seed, glm::uvec2 texture_size);

} // namespace raw::texture

#endif // SPACE_EXPLORER_GENERATOR_H
