//
// Created by progamers on 7/23/25.
//
#include <glm/glm.hpp>
namespace raw {
struct instance_data {
    glm::mat4 model;
    // ...

    static void setup_instance_attr(int starting_location);
};
} // namespace raw