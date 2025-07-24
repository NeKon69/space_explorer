//
// Created by progamers on 7/24/25.
//
#include "player_controller.h"

namespace raw {
player_controller::player_controller(raw::camera &cam) : camera(cam), velocity(1.0f) {}
void player_controller::update(const raw::movement_state &state, float delta_time) {
	glm::vec3 forward = camera.front();
	glm::vec3 right	  = camera.right();
	forward.y		  = 0;
	right.y			  = 0;
	if (glm::length(forward) > 0)
		forward = glm::normalize(forward);
	if (glm::length(right) > 0)
		right = glm::normalize(right);

	glm::vec3 desired_direction(0.0f);
	if (state.forward)
		desired_direction += forward;
	if (state.backward)
		desired_direction -= forward;
	if (state.left)
		desired_direction -= right;
	if (state.right)
		desired_direction += right;
	if (state.up)
		desired_direction += glm::vec3(0.0f, 1.0f, 0.0f);
	if (state.down)
		desired_direction -= glm::vec3(0.0f, 1.0f, 0.0f);

	if (glm::length(desired_direction) > 0.0f) {
		desired_direction = glm::normalize(desired_direction);
		velocity += desired_direction * predef::ACCELERATION * delta_time;
	} else {
		float speed = glm::length(velocity);
		if (speed > 0.01f) {
			glm::vec3 friction_force = -glm::normalize(velocity) * predef::FRICTION * delta_time;
			if (glm::length(friction_force) > speed) {
				velocity = glm::vec3(0.0f);
			} else {
				velocity += friction_force;
			}
		} else {
			velocity = glm::vec3(0.0f);
		}
	}
	if (glm::length(velocity) > predef::MAX_SPEED) {
		velocity = glm::normalize(velocity) * predef::MAX_SPEED;
	}
    camera.move(velocity * delta_time);
}
} // namespace raw