//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_SIMULATION_STATE_H
#define SPACE_EXPLORER_SIMULATION_STATE_H
#include "n_body/fwd.h"

namespace raw {
struct simulation_state {
	bool		 running;
	unsigned int amount_of_objects = 0;

	void add();
};
} // namespace raw

#endif // SPACE_EXPLORER_SIMULATION_STATE_H
