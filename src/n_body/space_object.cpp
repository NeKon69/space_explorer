//
// Created by progamers on 7/20/25.
//
#include "n_body/space_object.h"

#include "n_body/interaction_system.h"
#include "n_body/launch_leapfrog.h"

namespace raw {

space_object_data::space_object_data()
	: position(0.0),
	  velocity(predef::BASIC_VELOCITY),
	  mass(predef::PLANET_MASS),
	  radius(predef::BASIC_RADIUS) {}
space_object_data::space_object_data(glm::dvec3 _position, glm::dvec3 _velocity,
									 glm::dvec3 _acceleration, double _mass, double _radius)
	: position(_position),
	  velocity(_velocity),
	  acceleration(_acceleration),
	  mass(_mass),
	  radius(_radius) {}

space_object_data::space_object_data(const raw::space_object& object)
	: space_object_data(object.object_data) {}
space_object_data::space_object_data(raw::space_object* object)
	: space_object_data(object->object_data) {}

space_object::space_object() : object_data(glm::dvec3(1.0)) {}
space_object::space_object(glm::dvec3 _position, glm::dvec3 _velocity, glm::dvec3 _acceleration,
						   double _mass, double _radius)
	: object_data(_position, _velocity, _acceleration, _mass, _radius) {}
void space_object::update_position(space_object* data_first,
								   time since_last_upd, unsigned int count) {
	auto g = 1.0;
	launch_leapfrog(data_first, since_last_upd, count, g);
}
} // namespace raw