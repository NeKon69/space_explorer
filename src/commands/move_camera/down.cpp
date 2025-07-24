//
// Created by progamers on 7/24/25.
//
#include "commands/move_camera/down.h"

#include "scene.h"

namespace raw::command {
    void move_camera_down::execute(raw::scene& scene) {
        scene.move_camera_down();
    }
} // namespace raw::command