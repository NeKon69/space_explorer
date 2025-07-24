//
// Created by progamers on 7/24/25.
//
#include "commands/move_camera/up.h"

#include "scene.h"

namespace raw::command {
    void move_camera_up::execute(raw::scene& scene) {
        scene.move_camera_up();
    }
} // namespace raw::command