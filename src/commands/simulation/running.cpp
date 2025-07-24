//
// Created by progamers on 7/24/25.
//
#include "commands/simulation/running.h"
#include "scene.h"

namespace raw::command {
void change_simulation_running::execute(raw::scene &scene) {
	scene.change_sim_running_state();
}
} // namespace raw::command