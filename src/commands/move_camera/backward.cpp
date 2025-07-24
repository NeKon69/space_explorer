//
// Created by progamers on 7/24/25.
//
#include "commands/move_camera/backward.h"

#include "scene.h"

namespace raw::command {
    void move_camera_backward::execute(raw::scene& scene) {
        scene.move_camera_backward();
    }
} // namespace raw::command