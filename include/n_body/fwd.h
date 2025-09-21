//
// Created by progamers on 9/6/25.
//

#pragma once

#include <glm/glm.hpp>
namespace raw::n_body {
template<typename T>
class i_n_body_resource_manager;
template<typename T>
class i_n_body_simulator;
template<typename T>
class interaction_system;
namespace predef {
static constexpr auto BASIC_VELOCITY	 = glm::vec3(0.0f, 0.0f, 0.0f);
static constexpr auto BASIC_ACCELERATION = glm::vec3(0.0f);
static constexpr auto PLANET_MASS		 = 1.0;
static constexpr auto RADIUS			 = 1.0;
} // namespace predef
struct physics_component;
template<typename T>
struct space_object_data;
} // namespace raw::n_body
