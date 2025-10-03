//
// Created by progamers on 9/7/25.
//

#pragma once

#include <memory>

#include "core/clock.h"
#include "device_types/i_queue.h"
#include "graphics/gl_context_lock.h"
#include "n_body/fwd.h"
#include "n_body/i_n_body_resource_manager.h"
#include "n_body/i_n_body_simulator.h"
#include "n_body/n_body_predef.h"
namespace raw::n_body {
template<typename T>
class interaction_system {
private:
	std::shared_ptr<i_n_body_resource_manager<T>>	   resource_manager;
	std::shared_ptr<i_n_body_simulator<T>>			   simulator;
	std::shared_ptr<i_queue>						   queue;
	std::shared_ptr<entity_management::entity_manager> entity_manager;
	core::clock										   clock;
	graphics::graphics_data&						   graphics_data;
	double											   g	   = predef::G;
	double											   epsilon = 1;
	bool											   paused  = false;

public:
	interaction_system(std::shared_ptr<i_n_body_resource_manager<T>>	  resource_manager,
					   std::shared_ptr<i_n_body_simulator<T>>			  simulator,
					   const std::shared_ptr<i_queue>					  queue,
					   std::shared_ptr<entity_management::entity_manager> manager,
					   graphics::graphics_data&							  graphics_data)
		: resource_manager(resource_manager),
		  simulator(simulator),
		  queue(queue),
		  entity_manager(manager),
		  graphics_data(graphics_data) {}
	void update_simulation() {
		core::time time_ = clock.restart();
		if (!paused) {
			simulator->step(time_, queue, resource_manager, g, epsilon, graphics_data);
		}
	}
	void pause() {
		paused = true;
	}
	void resume() {
		paused = false;
	}
	[[nodiscard]] bool running() const {
		return !paused;
	}
	[[nodiscard]] uint32_t get_amount() const {
		return resource_manager->get_amount();
	}
	void sync() {
		simulator->sync();
	}
};
} // namespace raw::n_body
