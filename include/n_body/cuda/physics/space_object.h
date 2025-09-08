//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_SPACE_OBJECT_H
#define SPACE_EXPLORER_SPACE_OBJECT_H

#include <glm/glm.hpp>
namespace raw::n_body::cuda::physics {
namespace predef {
static constexpr auto BASIC_VELOCITY	 = glm::vec3(0.0f, 0.0f, 0.0f);
static constexpr auto BASIC_ACCELERATION = glm::vec3(0.0f);
static constexpr auto PLANET_MASS		 = 1.0;
static constexpr auto RADIUS			 = 1.0;
} // namespace predef

template<typename T>
struct space_object_data {
	glm::vec<3, T> position;
	glm::vec<3, T> velocity;
	T			   mass;
	T			   radius;
	uint32_t	   id;

	space_object_data()
		: position(0.0),
		  velocity(predef::BASIC_VELOCITY),
		  mass(predef::PLANET_MASS),
		  radius(predef::RADIUS) {}

	explicit space_object_data(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
							   double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS)
		: position(_position), velocity(_velocity), mass(_mass), radius(_radius) {}
};
} // namespace raw::n_body::cuda::physics

#endif // SPACE_EXPLORER_SPACE_OBJECT_H