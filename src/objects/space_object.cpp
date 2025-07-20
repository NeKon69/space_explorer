//
// Created by progamers on 7/20/25.
//
#include "objects/space_object.h"

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
	add_new_object();
}
space_object::space_object(glm::dvec3 _position, glm::dvec3 _velocity, double _mass, double _radius)
	: object_data(_position, _velocity, _mass, _radius) {
	add_new_object();
}
void space_object::update_position() {
	if (data_changed) {
		d_objects.allocate(sizeof(space_object_data) * c_objects.size());
	}
}
void space_object::add_new_object() const {
	c_objects.push_back(this->object_data);
	data_changed = true;
}

void new_pp(){
    space_object::c_objects = std::vector<space_object_data>(10);

}
} // namespace raw