//
// Created by progamers on 6/27/25.
//

#include <glm/glm.hpp>

#include "helper_macros.h"

#ifndef SPACE_EXPLORER_CAMERA_H
#define SPACE_EXPLORER_CAMERA_H

namespace raw {

namespace predef {
PASSIVE_VALUE CAMERA_POS	   = glm::vec3(0.0f, 0.0f, 5.0f);
PASSIVE_VALUE CAMERA_FRONT	   = glm::vec3(0.0f, 0.0f, -1.0f);
PASSIVE_VALUE CAMERA_UP		   = glm::vec3(0.0f, 1.0f, 0.0f);
PASSIVE_VALUE SENSITIVITY	   = 0.1f;
PASSIVE_VALUE CAMERA_SPEED	   = 0.05f;
PASSIVE_VALUE FOV			   = 45.0f;
PASSIVE_VALUE NEAR_PLANE	   = 0.1f;
PASSIVE_VALUE FAR_PLANE		   = 100.0f;
PASSIVE_VALUE WINDOW_WIDTH	   = 2560.f;
PASSIVE_VALUE WINDOW_HEIGHT	   = 1440.f;
PASSIVE_VALUE ASPECT_RATIO	   = WINDOW_WIDTH / WINDOW_HEIGHT;
PASSIVE_VALUE UPDATE_MOVE_TIME = 1000 / 10;
} // namespace predef

namespace camera_move {
void UP(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void DOWN(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void LEFT(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void RIGHT(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void FORWARD(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void BACKWARD(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
// Not defined, acts only as a type for function pointer
void MOVE_FUNCTION(glm::vec3&, glm::vec3, glm::vec3);
} // namespace camera_move

class camera {
private:
	glm::vec3 camera_pos;
	glm::vec3 camera_front;
	glm::vec3 camera_up;

	float yaw = -90., f, pitch = 0.f;
	float fov;
	// those need to be manually set (I have the setter for it), since I don't really want to
	// somehow make window and camera depend on each other
	float window_aspect_ratio = predef::ASPECT_RATIO;

public:
	// that constructor seems to be useless, since who in the right mind would create vec3's and
	// then pass it here, I think more logical would be to let the class handle all of it
	camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up);
	camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up, float _fov);
	camera();

	// basically same things just under different names
	[[nodiscard]] glm::mat4 view_projection() const;
	[[nodiscard]] glm::mat4 projection() const;
	[[nodiscard]] glm::mat4 view() const;

	/**
	 * @brief
	 * set new rotation for camera
	 * @param yaw absolute angle in degrees on y axis
	 * @param pitch absolute angle in degrees on x axis
	 */
	template<typename... Func>
	void set_rotation(float xoffset, float yoffset, Func&&... update_shader_uniforms) {
		yaw += xoffset;
		pitch -= yoffset;
		if (pitch > 89.0f) {
			pitch = 89.0f;
		}
		if (pitch < -89.0f) {
			pitch = -89.0f;
		}
		glm::vec3 front;
		front.x		 = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
		front.y		 = sinf(glm::radians(pitch));
		front.z		 = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
		camera_front = glm::normalize(front);
		// That's just a check for me, so I don't forget to update shader uniforms
		static_assert(sizeof...(Func) != 0,
					  "You must provide at least one function to update shader uniforms");
		(update_shader_uniforms(), ...);
	}

	// again I come up with some stupid and at the same time genius ideas
	/**
	 * @brief
	 * move camera by the function parameter(you can pass any function specified in raw::camera_move
	 * namespace, or add one yourself)
	 * @param func function from raw::camera_move namespace
	 */
	template<typename... Func>
	void move(decltype(camera_move::MOVE_FUNCTION) func, Func&&... update_shader_uniforms) {
		func(camera_pos, camera_front, camera_up);
		// That's just a check for me, so I don't forget to update shader uniforms
		static_assert(sizeof...(Func) != 0,
					  "You must provide at least one function to update shader uniforms");
		(update_shader_uniforms(), ...);
	}

	template<typename... Func>
	void adjust_fov(float delta, Func&&... update_shader_uniforms) {
		fov += delta;
		if (fov < 1.0f) {
			fov = 1.0f;
		}
		if (fov > 180.0f) {
			fov = 180.0f;
		}
		// That's just a check for me, so I don't forget to update shader uniforms
		static_assert(sizeof...(Func) != 0,
					  "You must provide at least one function to update shader uniforms");
		(update_shader_uniforms(), ...);
	}

	[[nodiscard]] inline glm::vec3 pos() const {
		return camera_pos;
	}
	[[nodiscard]] inline glm::vec3 front() const {
		return camera_front;
	}
	[[nodiscard]] inline glm::vec3 up() const {
		return camera_up;
	}

	void inline set_window_resolution(int x, int y) {
		window_aspect_ratio = static_cast<float>(x) / static_cast<float>(y);
	}
};

} // namespace raw

#endif // SPACE_EXPLORER_CAMERA_H
