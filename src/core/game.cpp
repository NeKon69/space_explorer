//
// Created by progamers on 7/24/25.
//
#include "core/game.h"

#include "game_states/playing_state.h"

namespace raw::core {
/**
 * @brief Constructs a game and initializes its renderer and initial state.
 *
 * Initializes the renderer using the provided window/application name and
 * pushes a new playing_state (constructed with the renderer's rendering data
 * and current window size) onto the internal state stack.
 *
 * @param name Window or application name passed to the renderer.
 */
game::game(const std::string &name) : renderer(name) {
	states.emplace(std::make_unique<game_states::playing_state>(renderer->get_data(),
																renderer->get_window_size()));
}

/**
 * @brief Runs the main game loop until no states remain or the game is stopped.
 *
 * This executes the frame loop: it measures per-frame delta time, lets the current
 * top state handle input (which may signal that it should be popped), updates the
 * active top state with the measured delta time, and instructs it to draw using
 * the game's renderer. If the state stack becomes empty at any point, the loop
 * ends and the game's running flag is cleared.
 *
 * Side effects:
 * - May call pop_state() when a state signals it should be removed.
 * - May set the game's is_running flag to false when there are no states.
 */
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

/**
 * @brief Remove the top state from the state's stack.
 *
 * Removes the current (top) state from the internal stack of game states.
 *
 * @note Precondition: the state stack must be non-empty before calling this method;
 * calling this when the stack is empty results in undefined behavior.
 */
void game::pop_state() {
	states.pop();
}
} // namespace raw::core