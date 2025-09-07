//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_INTERACTION_SYSTEM_H
#define SPACE_EXPLORER_INTERACTION_SYSTEM_H
#include <memory>

#include "core/clock.h"
#include "n_body/fwd.h"
namespace raw::n_body {
template<typename T>
class interaction_system {
private:
	std::shared_ptr<i_n_body_resource_manager<T>> resource_manager;
	std::shared_ptr<i_n_body_simulator<T>>		  simulator;
	core::clock									  clock;
	bool										  paused = false;

public:
	void update_simulation();
	void pause();
	void resume();
	// TODO: add some other methods like "add_object" and so on
};
} // namespace raw::n_body
#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
