//
// Created by progamers on 7/23/25.
//
#include "instance_data.h"

#include <glad/glad.h>
namespace raw {
void instance_data::setup_instance_attr(int starting_location) {
	// For now here is only setup to instance mat4 model, later I will add textures, maps, ect...
	glVertexAttribPointer(starting_location, 4, GL_FLOAT, GL_FALSE, sizeof(model), nullptr);
	glEnableVertexAttribArray(starting_location);
	glVertexAttribDivisor(starting_location++, 0);

	glVertexAttribPointer(starting_location, 4, GL_FLOAT, GL_FALSE, sizeof(model),
						  (void*)(sizeof(model) / 4));
	glEnableVertexAttribArray(starting_location);
	glVertexAttribDivisor(starting_location++, 1);

	glVertexAttribPointer(starting_location, 4, GL_FLOAT, GL_FALSE, sizeof(model),
						  (void*)(sizeof(model) / 2));
	glEnableVertexAttribArray(starting_location);
	glVertexAttribDivisor(starting_location++, 1);

	glVertexAttribPointer(starting_location, 4, GL_FLOAT, GL_FALSE, sizeof(model),
						  (void*)(sizeof(model) / 4 * 3));
	glEnableVertexAttribArray(starting_location);
	glVertexAttribDivisor(starting_location++, 1);
}
} // namespace raw
