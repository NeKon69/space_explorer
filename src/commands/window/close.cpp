//
// Created by progamers on 7/24/25.
//
#include "commands/window/close.h"
#include "scene.h"

namespace raw::command {
    void window_close::execute(raw::scene &scene) {
        scene.close_window();
    }
} // namespace raw::command