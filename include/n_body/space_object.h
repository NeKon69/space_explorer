//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_SPACE_OBJECT_H
#define SPACE_EXPLORER_SPACE_OBJECT_H
#include <glm/glm.hpp>

#include "clock.h"
#include "cuda_types/buffer.h"
#include "objects/sphere.h"
namespace raw {
namespace predef {
PASSIVE_VALUE BASIC_VELOCITY	 = glm::vec3(0.0f, 0.0f, 0.0f);
PASSIVE_VALUE BASIC_ACCELERATION = glm::vec3(0.0f);
PASSIVE_VALUE PLANET_MASS		 = 1.0;
PASSIVE_VALUE RADIUS			 = 1.0;
} // namespace predef

class space_object;

// FIXME: Since as far as I know, gpu sucks at double precision float, it would be better to make
// all things related to simulation templated, so I cau use float more (my gpu is slower on double
// for almost 5 times)
struct space_object_data {
	glm::dvec3 position;
	glm::dvec3 velocity;
	double	   mass;
	double	   radius;
	space_object_data();
	explicit space_object_data(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
							   double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS);
	explicit space_object_data(const space_object& object);
	explicit space_object_data(space_object* object);
	space_object_data(const space_object_data&) = default;
};

class interaction_system;
class drawable_space_object;
class space_object {
private:
	space_object_data object_data;
	friend class drawable_space_object;
	friend struct space_object_data;
	friend class interaction_system;

public:
	space_object();
	explicit space_object(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
						  double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS);
	static void update_position(space_object* data_first, time since_last_upd, unsigned int count);
	__device__ __host__ space_object_data& get() {
		return object_data;
	}
	space_object(const space_object& object)			= default;
	space_object& operator=(const space_object& object) = default;
};

} // namespace raw

#endif // SPACE_EXPLORER_SPACE_OBJECT_H
