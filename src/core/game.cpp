//
// Created by progamers on 7/24/25.
//
#include "core/game.h"

#include "game_states/playing_state.h"

namespace raw::core {
game::game(const std::string &name) : renderer(name) {
	states.emplace(std::make_unique<game_states::playing_state>(renderer->get_window_size()));
}

void game::run() {
	raw::core::clock clock;
	while (is_running) {
		auto delta_time = clock.restart();
		delta_time.to_milli();
		if (states.empty()) {
			is_running = false;
			continue;
		}

		// returned true, means we gotta pop it from the stack
		if (states.top()->handle_input()) {
			pop_state();
			if (states.empty()) {
				is_running = false;
				continue;
			}
		}
		states.top()->update(delta_time);
		states.top()->draw(renderer);
	}
}

void game::pop_state() {
	states.pop();
}
} // namespace raw::core