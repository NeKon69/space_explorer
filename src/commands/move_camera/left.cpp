//
// Created by progamers on 7/24/25.
//
#include "commands/move_camera/left.h"

#include "scene.h"

namespace raw::command {
void move_camera_left::execute(raw::scene& scene) {
	scene.move_camera_left();
}
} // namespace raw::command