//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include <array>
#include <glm/glm.hpp>
namespace raw {
extern constexpr std::array<glm::vec3, 12> generate_icosahedron(float radius);
} // namespace raw
#endif // SPACE_EXPLORER_MESH_GENERATOR_H
