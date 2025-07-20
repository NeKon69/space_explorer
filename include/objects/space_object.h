//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_SPACE_OBJECT_H
#define SPACE_EXPLORER_SPACE_OBJECT_H
#include <glm/glm.hpp>

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

class space_object {
public:
	// The data for n-body
	static cuda_buffer<space_object_data> d_objects;
	static std::vector<space_object_data> c_objects;
	static bool							  data_changed;
	// Idk let's keep the fucking how to draw this object here, inheritance won't work, since then i
	// would have fucking trillion billion data consumed on gpu, which we don't like
	static sphere	  structure;
private:
	space_object_data object_data;

	void add_new_object() const;

public:
	space_object();
	explicit space_object(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
						  double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS);
	void update_position();
	void draw();
};

} // namespace raw

#endif // SPACE_EXPLORER_SPACE_OBJECT_H
