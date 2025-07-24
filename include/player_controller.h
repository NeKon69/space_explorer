//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_PLAYER_CONTROLLER_H
#define SPACE_EXPLORER_PLAYER_CONTROLLER_H
#include <glm/glm.hpp>

#include "camera.h"
#include "movement_state.h"
namespace raw {

namespace predef {
PASSIVE_VALUE ACCELERATION = 30.0f;
PASSIVE_VALUE MAX_SPEED	   = 20.0f;
PASSIVE_VALUE FRICTION	   = 8.0f;
} // namespace predef

class player_controller {
private:
	raw::camera& camera;
	glm::vec3	 velocity;

public:
	explicit player_controller(raw::camera& cam);
    void update(const raw::movement_state& state, float delta_time);
};
} // namespace raw
#endif // SPACE_EXPLORER_PLAYER_CONTROLLER_H
