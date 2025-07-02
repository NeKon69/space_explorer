//
// Created by progamers on 6/27/25.
//

#include <glm/glm.hpp>
#include "helper_macros.h"

#ifndef SPACE_EXPLORER_CAMERA_H
#define SPACE_EXPLORER_CAMERA_H

namespace raw {

namespace predef {
PASSIVE_VALUE CAMERA_POS   = glm::vec3(0.0f, 0.0f, 5.0f);
PASSIVE_VALUE CAMERA_FRONT = glm::vec3(0.0f, 0.0f, -1.0f);
PASSIVE_VALUE CAMERA_UP	   = glm::vec3(0.0f, 1.0f, 0.0f);
PASSIVE_VALUE SENSITIVITY  = 0.1f;
PASSIVE_VALUE CAMERA_SPEED = 0.05f;
} // namespace predef

namespace camera_move {
void UP(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void DOWN(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void LEFT(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void RIGHT(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void FORWARD(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void BACKWARD(glm::vec3& pos, glm::vec3 front, glm::vec3 up);
void MOVE_FUNCTION(glm::vec3&, glm::vec3, glm::vec3);
} // namespace camera_move

class camera {
private:
	glm::vec3 camera_pos;
	glm::vec3 camera_front;
	glm::vec3 camera_up;

public:
	// that constructor seems to be useless, since who in the right mind would create vec3's and
	// then pass it here, I think more logical would be to let the class handle all of it
	camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up);
	explicit camera(glm::vec3 vec);
	camera();

	// basically same things just under different names
	[[nodiscard]] glm::mat4 value()const ;
    [[nodiscard]] glm::mat4 operator()()const;
    [[nodiscard]] glm::mat4 update()const;

	void rotate(float yaw, float pitch);
	void move(decltype(camera_move::MOVE_FUNCTION) func);

    [[nodiscard]] inline glm::vec3 pos() const {
		return camera_pos;
	}
    [[nodiscard]] inline glm::vec3 front() const {
		return camera_front;
	}
    [[nodiscard]] inline glm::vec3 up() const {
		return camera_up;
	}
};

} // namespace raw

#endif // SPACE_EXPLORER_CAMERA_H
