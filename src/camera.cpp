//
// Created by progamers on 6/27/25.
//
#include "../include/core/camera/camera.h"

#include <glm/gtc/matrix_transform.hpp>

namespace raw {
    void camera_move::DOWN(glm::vec3 &pos, glm::vec3, glm::vec3) {
        pos.y -= predef::CAMERA_SPEED;
    }

    void camera_move::UP(glm::vec3 &pos, glm::vec3, glm::vec3) {
        pos.y += predef::CAMERA_SPEED;
    }

    void camera_move::LEFT(glm::vec3 &pos, glm::vec3 front, glm::vec3 up) {
        pos -= glm::normalize(glm::cross(front, up)) * raw::predef::CAMERA_SPEED;
    }

    void camera_move::RIGHT(glm::vec3 &pos, glm::vec3 front, glm::vec3 up) {
        pos += glm::normalize(glm::cross(front, up)) * raw::predef::CAMERA_SPEED;
    }

    void camera_move::FORWARD(glm::vec3 &pos, glm::vec3 front, glm::vec3) {
        pos += front * raw::predef::CAMERA_SPEED;
    }

    void camera_move::BACKWARD(glm::vec3 &pos, glm::vec3 front, glm::vec3) {
        pos -= front * raw::predef::CAMERA_SPEED;
    }

    camera::camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up)
        : camera_pos(_pos), camera_front(_front), camera_up(_up), fov(predef::FOV) {
    }

    camera::camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up, float _fov)
        : camera_pos(_pos), camera_front(_front), camera_up(_up), fov(_fov) {
    }

    camera::camera()
        : camera_pos(predef::CAMERA_POS),
          camera_front(predef::CAMERA_FRONT),
          camera_up(predef::CAMERA_UP),
          fov(predef::FOV) {
    }

    glm::mat4 camera::view_projection() const {
        return projection() * view();
    }

    glm::mat4 camera::projection() const {
        return glm::perspective(glm::radians(fov), window_aspect_ratio, predef::NEAR_PLANE,
                                predef::FAR_PLANE);
    }

    void camera::move(const glm::vec3 &offset) {
        camera_pos += offset;
    };

    glm::mat4 camera::view() const {
        return glm::lookAt(camera_pos, camera_front + camera_pos, camera_up);
    }
} // namespace raw