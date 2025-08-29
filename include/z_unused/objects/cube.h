//
// Created by progamers on 7/1/25.
//

#ifndef SPACE_EXPLORER_CUBE_H
#define SPACE_EXPLORER_CUBE_H
#include <glm/glm.hpp>

#include "../z_unused/object.h"

namespace raw::z_unused::objects {
namespace axis {
inline constexpr auto X = glm::vec3(1.0, 0.0, 0.0);
inline constexpr auto Y = glm::vec3(0.0, 1.0, 0.0);
inline constexpr auto Z = glm::vec3(0.0, 0.0, 1.0);
} // namespace axis

class cube : public object {
private:
	// static is for define it as class member not as instance of this class member. inline is for
	// say to compiler that he should not create a new copy of this variable in each translation
	// unit, but rather use the same one.
	static inline constexpr auto cube_pos = {
		-0.5f, -0.5f, -0.5f, 0.0f,	0.0f,  -1.0f, 0.5f,	 -0.5f, -0.5f, 0.0f,  0.0f,	 -1.0f,
		0.5f,  0.5f,  -0.5f, 0.0f,	0.0f,  -1.0f, -0.5f, 0.5f,	-0.5f, 0.0f,  0.0f,	 -1.0f,

		-0.5f, -0.5f, 0.5f,	 0.0f,	0.0f,  1.0f,  0.5f,	 -0.5f, 0.5f,  0.0f,  0.0f,	 1.0f,
		0.5f,  0.5f,  0.5f,	 0.0f,	0.0f,  1.0f,  -0.5f, 0.5f,	0.5f,  0.0f,  0.0f,	 1.0f,

		-0.5f, 0.5f,  0.5f,	 -1.0f, 0.0f,  0.0f,  -0.5f, 0.5f,	-0.5f, -1.0f, 0.0f,	 0.0f,
		-0.5f, -0.5f, -0.5f, -1.0f, 0.0f,  0.0f,  -0.5f, -0.5f, 0.5f,  -1.0f, 0.0f,	 0.0f,

		0.5f,  0.5f,  0.5f,	 1.0f,	0.0f,  0.0f,  0.5f,	 0.5f,	-0.5f, 1.0f,  0.0f,	 0.0f,
		0.5f,  -0.5f, -0.5f, 1.0f,	0.0f,  0.0f,  0.5f,	 -0.5f, 0.5f,  1.0f,  0.0f,	 0.0f,

		-0.5f, -0.5f, -0.5f, 0.0f,	-1.0f, 0.0f,  0.5f,	 -0.5f, -0.5f, 0.0f,  -1.0f, 0.0f,
		0.5f,  -0.5f, 0.5f,	 0.0f,	-1.0f, 0.0f,  -0.5f, -0.5f, 0.5f,  0.0f,  -1.0f, 0.0f,

		-0.5f, 0.5f,  -0.5f, 0.0f,	1.0f,  0.0f,  0.5f,	 0.5f,	-0.5f, 0.0f,  1.0f,	 0.0f,
		0.5f,  0.5f,  0.5f,	 0.0f,	1.0f,  0.0f,  -0.5f, 0.5f,	0.5f,  0.0f,  1.0f,	 0.0f};
	// same applies here
	static inline constexpr auto indices = {0,	1,	2,	2,	3,	0,	4,	5,	6,	6,	7,	4,
											8,	9,	10, 10, 11, 8,	12, 13, 14, 14, 15, 12,
											16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};
	// code above is ugly, but where else to put it?

public:
	// that class is complete trash now haha, it makes sense actually that object hold shader and
	// transformation matrix, since you apply to the OBJECT those transformations not to the cube
	// specifically
	using object::object;

	explicit cube(const std::shared_ptr<raw::rendering::shader::shader> &sh);
};
} // namespace raw::z_unused::objects

#endif // SPACE_EXPLORER_CUBE_H