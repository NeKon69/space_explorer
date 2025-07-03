//
// Created by progamers on 6/27/25.
//
#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>

namespace raw {

void camera_move::DOWN(glm::vec3& pos, glm::vec3, glm::vec3) {
	pos.y -= predef::CAMERA_SPEED;
}
void camera_move::UP(glm::vec3& pos, glm::vec3, glm::vec3) {
	pos.y += predef::CAMERA_SPEED;
}
void camera_move::LEFT(glm::vec3& pos, glm::vec3 front, glm::vec3 up) {
	pos -= glm::normalize(glm::cross(front, up)) * raw::predef::CAMERA_SPEED;
}
void camera_move::RIGHT(glm::vec3& pos, glm::vec3 front, glm::vec3 up) {
	pos += glm::normalize(glm::cross(front, up)) * raw::predef::CAMERA_SPEED;
}
void camera_move::FORWARD(glm::vec3& pos, glm::vec3 front, glm::vec3) {
	pos += front * raw::predef::CAMERA_SPEED;
}
void camera_move::BACKWARD(glm::vec3& pos, glm::vec3 front, glm::vec3) {
    pos -= front * raw::predef::CAMERA_SPEED;
}

camera::camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up)
	: camera_pos(_pos), camera_front(_front), camera_up(_up) {}
camera::camera()
	: camera_pos(predef::CAMERA_POS),
	  camera_front(predef::CAMERA_FRONT),
	  camera_up(predef::CAMERA_UP) {}

glm::mat4 camera::value() const {
	return glm::lookAt(camera_pos, camera_front + camera_pos, camera_up);
}
glm::mat4 camera::operator()() const {
	return value();
}
glm::mat4 camera::update() const {
	return value();
}

void camera::rotate(float yaw, float pitch) {
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
}

// again I come up with some stupid and at the same time genius ideas
void camera::move(decltype(camera_move::MOVE_FUNCTION) func) {
	func(camera_pos, camera_front, camera_up);
}
} // namespace raw