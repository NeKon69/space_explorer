//
// Created by progamers on 7/1/25.
//
#include "z_unused/objects/cube.h"

namespace raw::z_unused::objects {
cube::cube(const raw::shared_ptr<raw::rendering::shader::shader> &sh) : object(cube_pos, indices) {
	this->shader = sh;
}
} // namespace raw::z_unused::objects