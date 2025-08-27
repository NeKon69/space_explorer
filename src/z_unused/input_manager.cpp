//
// Created by progamers on 7/24/25.
//
#include "z_unused/input_manager.h"

#include <SDL3/SDL.h>

namespace raw {
	void input_manager::init() {
		// Unfortunately I did not make constructor from inherited type, so we are forced to use std
		// smart pointers
		//	key_bindings[SDL_SCANCODE_W]	  = std::make_shared<raw::command::move_camera_forward>();
		//	key_bindings[SDL_SCANCODE_A]	  = std::make_shared<raw::command::move_camera_left>();
		//	key_bindings[SDL_SCANCODE_S]	  = std::make_shared<raw::command::move_camera_backward>();
		//	key_bindings[SDL_SCANCODE_D]	  = std::make_shared<raw::command::move_camera_right>();
		//	key_bindings[SDL_SCANCODE_SPACE]  = std::make_shared<raw::command::move_camera_up>();
		//	key_bindings[SDL_SCANCODE_TAB]	  = std::make_shared<raw::command::move_camera_down>();
		//	key_bindings[SDL_SCANCODE_ESCAPE] = std::make_shared<raw::command::window_close>();
		//	key_bindings[SDL_SCANCODE_T] =
		//std::make_shared<raw::command::directional_light_change_state>(); 	key_bindings[SDL_SCANCODE_P]
		//= std::make_shared<raw::command::change_simulation_running>();
	}

	void input_manager::handle_events() {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (key_bindings.contains(event.key.scancode)) {
				command_queue.emplace_back(key_bindings[event.key.scancode]);
			}
		}
	}

	inline std::vector<std::shared_ptr<raw::command::command> > &&input_manager::get_commands() {
		return std::move(command_queue);
	}
} // namespace raw
