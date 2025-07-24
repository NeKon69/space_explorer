//
// Created by progamers on 7/24/25.
//
#include "commands/directional_light/change_state.h"

#include "scene.h"

namespace raw::command {

void directional_light_change_state::execute(raw::scene& scene) {
	scene.change_dir_light_state();
}

} // namespace raw::command