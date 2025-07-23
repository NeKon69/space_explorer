//
// Created by progamers on 7/23/25.
//
#include <glm/glm.hpp>

#include "helper_macros.h"
namespace raw {
namespace predef {
    PASSIVE_VALUE AMOUNT_OF_INSTANCES = 1000;
}
struct instance_data {
	glm::mat4 model;
	// ...

	static void setup_instance_attr(int starting_location);
};
} // namespace raw