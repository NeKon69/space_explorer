//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_GENERATOR_H
#define SPACE_EXPLORER_GENERATOR_H
#include <glm/glm.hpp>

#include "common/fwd.h"
#include "textures/fwd.h"

namespace raw::textures {
    // Acts as a host side launcher
    // Also does some checks to not compile if expected "output_data" byte size doesn't match the byte
    // size expected
    void generate(raw::textures::data output_data, raw::textures::planet_dna planet_data,
                  raw::UI seed, glm::uvec2 texture_size);
} // namespace raw::textures

#endif // SPACE_EXPLORER_GENERATOR_H