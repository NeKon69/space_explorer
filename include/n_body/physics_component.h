//
// Created by progamers on 9/19/25.
//

#pragma once
#include "components/component.h"
#include "n_body/fwd.h"

namespace raw::n_body {
struct physics_component : components::component {};

template<typename T>
struct space_object_data {
	glm::vec<3, T> position;
	glm::vec<3, T> velocity;
	T			   mass;
	T			   radius;

	constexpr space_object_data()
		: position(0.0),
		  velocity(predef::BASIC_VELOCITY),
		  mass(predef::PLANET_MASS),
		  radius(predef::RADIUS) {}

	explicit constexpr space_object_data(glm::dvec3 _position,
										 glm::dvec3 _velocity = predef::BASIC_VELOCITY,
										 double		_mass	  = predef::PLANET_MASS,
										 double		_radius	  = predef::RADIUS)
		: position(_position), velocity(_velocity), mass(_mass), radius(_radius) {}

	space_object_data(space_object_data&& other)				 = default;
	space_object_data& operator=(space_object_data&& other)		 = default;
	space_object_data(const space_object_data& other)			 = default;
	space_object_data& operator=(const space_object_data& other) = default;
};
} // namespace raw::n_body