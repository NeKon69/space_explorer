//
// Created by progamers on 8/3/25.
//

#ifndef SPACE_EXPLORER_VERTEX_H
#define SPACE_EXPLORER_VERTEX_H
#include <glm/glm.hpp>
#include "common/fwd.h"

namespace raw::graphics {
	struct vertex {
		glm::vec3 position;
		glm::vec2 tex_coord;
		// Normal mapping
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec3 bitangent;
	};
} // namespace raw
#endif // SPACE_EXPLORER_VERTEX_H