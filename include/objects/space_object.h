//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_SPACE_OBJECT_H
#define SPACE_EXPLORER_SPACE_OBJECT_H
#include <glm/glm.hpp>
#include "clock.h"
#include "cuda_types/buffer.h"
#include "sphere.h"
namespace raw {
namespace predef {
PASSIVE_VALUE BASIC_VELOCITY = glm::vec3(5.0f, 5.0f, 5.0f);
PASSIVE_VALUE PLANET_MASS	 = 1.0;
PASSIVE_VALUE RADIUS		 = 1.0;
} // namespace predef
struct space_object_data {
	glm::dvec3 position;
	glm::dvec3 velocity;
	double	   mass;
	double	   radius;
	space_object_data();
	explicit space_object_data(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
							   double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS);
};

class interaction_system;
class drawable_space_object;
class space_object {
private:
	space_object_data object_data;
	friend class drawable_space_object;

public:
	space_object();
	explicit space_object(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
						  double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS);
	void update_position(const interaction_system& system, time since_last_upd);
	space_object(const space_object& object)			= default;
	space_object& operator=(const space_object& object) = default;
};

} // namespace raw

#endif // SPACE_EXPLORER_SPACE_OBJECT_H
