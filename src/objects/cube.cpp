//
// Created by progamers on 7/1/25.
//
#include "objects/cube.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace raw {

cube::cube(raw::shared_ptr<raw::shader> sh) : object(cube_pos, indices) {
    this->shader = sh;
}

} // namespace raw