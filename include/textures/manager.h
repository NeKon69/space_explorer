//
// Created by progamers on 8/5/25.
//

#pragma once
#include "textures/fwd.h"

namespace raw::textures {
// For now, it's just an abstraction from getting seed for the planet to generate, but maybe later
// it will get some other use case
class manager {
private:
	planet_id current_seed = 0;

public:
	manager() = default;

	manager(planet_id starting_seed);

	/**
	 * @brief Increments the counter and returns new seed (that avoids problem in which user can
	 * just click generate infinitely and get same result)
	 */
	planet_id get_seed();
};
} // namespace raw::textures
