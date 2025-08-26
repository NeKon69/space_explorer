//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_PLAYER_CONTROLLER_H
#define SPACE_EXPLORER_PLAYER_CONTROLLER_H
#include <glm/glm.hpp>

#include "core/fwd.h"

namespace raw::core::camera {
namespace predef {
static constexpr auto ACCELERATION = 20.0f;
static constexpr auto MAX_SPEED	   = 10.0f;
static constexpr auto FRICTION	   = 30.0f;
} // namespace predef

class player_controller {
private:
	camera	 &camera;
	glm::vec3 velocity;

public:
	explicit player_controller(class camera &cam);

	void update(const core::camera::movement_state &state, float delta_time);
};
} // namespace raw::core::camera
#endif // SPACE_EXPLORER_PLAYER_CONTROLLER_H