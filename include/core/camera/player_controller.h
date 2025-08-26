//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_PLAYER_CONTROLLER_H
#define SPACE_EXPLORER_PLAYER_CONTROLLER_H
#include <glm/glm.hpp>

#include "core/fwd.h"
#include "helper/helper_macros.h"

namespace raw {

namespace predef {
PASSIVE_VALUE ACCELERATION = 20.0f;
PASSIVE_VALUE MAX_SPEED	   = 10.0f;
PASSIVE_VALUE FRICTION	   = 30.0f;
} // namespace predef

class player_controller {
private:
	raw::core::camera &camera;
	glm::vec3	 velocity;

public:
	explicit player_controller(raw::core::camera &cam);

	void update(const raw::core::camera_move::movement_state &state, float delta_time);
};
} // namespace raw
#endif // SPACE_EXPLORER_PLAYER_CONTROLLER_H
