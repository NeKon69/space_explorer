//
// Created by progamers on 7/6/25.
//
#include "../include/z_unused/object_info.h"

#include <glm/gtc/matrix_transform.hpp>

namespace raw {
    object_info::object_info(raw::shared_ptr<raw::object> obj) : object(obj) {
    }

    object_info::object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position)
        : object(obj), transform(glm::translate(transform, position)) {
    }

    object_info::object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position, glm::vec3 scale)
        : object(obj), transform(glm::translate(glm::scale(transform, position), scale)) {
    }

    object_info::object_info(raw::shared_ptr<raw::object> obj, glm::mat4 transformation)
        : object(obj), transform(transformation) {
    }

    object_info::object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position, glm::vec3 scale,
                             glm::vec3 rotation, float degree)
        : transform(glm::translate(
            glm::rotate(glm::scale(transform, scale), glm::radians(degree), rotation), position)) {
    }
} // namespace raw