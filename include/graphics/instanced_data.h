//
// Created by progamers on 9/6/25.
//

#ifndef SPACE_EXPLORER_INSTANCED_DATA_H
#define SPACE_EXPLORER_INSTANCED_DATA_H
#include <glm/glm.hpp>
namespace raw::graphics {
struct instanced_data {
	glm::mat4 model;
	uint64_t  texture1;
	uint64_t  texture2;
};
} // namespace raw::graphics
#endif // SPACE_EXPLORER_INSTANCED_DATA_H
