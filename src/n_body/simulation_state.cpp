//
// Created by progamers on 7/24/25.
//
#include "../../include/n_body/cuda/simulation_state.h"

namespace raw {
void simulation_state::add() {
	++amount_of_objects;
}
} // namespace raw