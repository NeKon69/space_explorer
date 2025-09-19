//
// Created by progamers on 9/19/25.
//

#pragma once
#include "components/component.h"
#include "n_body/fwd.h"

namespace raw::n_body {
template<typename T>
struct physics_component : components::component {
	glm::vec<3, T> position;
	glm::vec<3, T> velocity;
	T			   mass;
	T			   radius;

	HOST_DEVICE constexpr physics_component()
		: position(0.0),
		  velocity(predef::BASIC_VELOCITY),
		  mass(predef::PLANET_MASS),
		  radius(predef::RADIUS) {}

	HOST_DEVICE explicit constexpr physics_component(glm::dvec3 _position,
													 glm::dvec3 _velocity = predef::BASIC_VELOCITY,
													 double		_mass	  = predef::PLANET_MASS,
													 double		_radius	  = predef::RADIUS)
		: position(_position), velocity(_velocity), mass(_mass), radius(_radius) {}

	physics_component(physics_component&&)				   = default;
	physics_component& operator=(physics_component&&)	   = default;
	physics_component(const physics_component&)			   = default;
	physics_component& operator=(const physics_component&) = default;
};
} // namespace raw::n_body