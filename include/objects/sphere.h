//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_SPHERE_H
#define SPACE_EXPLORER_SPHERE_H
#include "object.h"
#include "sphere_generation/mesh_generator.h"

namespace raw {
namespace predef {
static const int MAXIMUM_AMOUNT_OF_INDICES =
	BASIC_AMOUNT_OF_TRIANGLES * static_cast<UI>(std::pow(4, MAX_STEPS)) + 2;
// It's actually 10, but I'll try and see if my algorithm fails on ten, for now, lets keep it at 20
static const int MAXIMUM_AMOUNT_OF_VERTICES =
	BASIC_AMOUNT_OF_TRIANGLES * static_cast<UI>(std::pow(4, MAX_STEPS));
} // namespace predef
class sphere : object {
private:
	icosahedron_generator gen;
	std::vector<UI>		  indices(predef::MAXIMUM_AMOUNT_OF_INDICES);
	std::vector<float>	  vertices(predef::MAXIMUM_AMOUNT_OF_VERTICES);

public:
	sphere(raw::shader& shader);
};

} // namespace raw
#endif // SPACE_EXPLORER_SPHERE_H
