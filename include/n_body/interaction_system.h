//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_INTERACTION_SYSTEM_H
#define SPACE_EXPLORER_INTERACTION_SYSTEM_H
#include <memory>

#include "core/clock.h"
#include "graphics/gl_context_lock.h"
#include "n_body/cuda/n_body_resource_manager.h"
#include "n_body/fwd.h"
#include "n_body/n_body_predef.h"
namespace raw::n_body {
template<typename T>
class interaction_system {
private:
	std::shared_ptr<i_n_body_resource_manager<T>> resource_manager;
	std::shared_ptr<i_n_body_simulator<T>>		  simulator;
	std::shared_ptr<i_queue>					  queue;
	core::clock									  clock;
	graphics::graphics_data&					  graphics_data;
	double										  g		  = predef::G;
	double										  epsilon = 1;
	bool										  paused  = false;

public:
	interaction_system(std::shared_ptr<i_n_body_resource_manager<T>> resource_manager,
					   std::shared_ptr<i_n_body_simulator<T>>		 simulator,
					   const std::shared_ptr<i_queue> queue, graphics::graphics_data& graphics_data)
		: resource_manager(resource_manager),
		  simulator(simulator),
		  queue(queue),
		  graphics_data(graphics_data) {}
	void update_simulation() {
		if (!paused) {
			simulator->step(clock.restart(), queue, resource_manager, g, epsilon, graphics_data);
		}
	}
	void pause() {
		paused = true;
	}
	void resume() {
		paused = false;
	}
	void add(space_object_data<T> object) {
		resource_manager->add(object);
	}
	// TODO: add some other methods like "add_object" and so on
};
} // namespace raw::n_body
#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
