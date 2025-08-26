//
// Created by progamers on 7/1/25.
//
#include "z_unused/objects/cube.h"

namespace raw {
    cube::cube(const raw::shared_ptr<raw::shader> &sh) : object(cube_pos, indices) {
        this->shader = sh;
    }
} // namespace raw