//
// Created by progamers on 7/24/25.
//
#include "commands/move_camera/forward.h"

#include "scene.h"

namespace raw::command {
void move_camera_forward::execute(raw::scene& scene) {
	scene.move_camera_right();
}
} // namespace raw::command