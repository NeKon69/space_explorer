//
// Created by progamers on 7/20/25.
//
#include "objects/space_object.h"
#include "objects/interaction_system.h"

namespace raw {


space_object_data::space_object_data()
	: position(0.0),
	  velocity(predef::BASIC_VELOCITY),
	  mass(predef::PLANET_MASS),
	  radius(predef::BASIC_RADIUS) {}
space_object_data::space_object_data(glm::dvec3 _position, glm::dvec3 _velocity, double _mass,
									 double _radius)
	: position(_position), velocity(_velocity), mass(_mass), radius(_radius) {}

space_object::space_object() : object_data(glm::dvec3(1.0)) {
}
space_object::space_object(glm::dvec3 _position, glm::dvec3 _velocity, double _mass, double _radius)
	: object_data(_position, _velocity, _mass, _radius) {
}
void space_object::update_position(const interaction_system& system, time since_last_upd) {
}

} // namespace raw