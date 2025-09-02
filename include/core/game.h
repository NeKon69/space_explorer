//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_GAME_H
#define SPACE_EXPLORER_GAME_H

#include <memory>
#include <stack>

#include "core/fwd.h"
#include "game_state.h"
#include "rendering/renderer.h"

/**
 * Construct a game instance with the given name.
 *
 * The name is used to identify the game (e.g., window title or project identifier).
 * @param name Human-readable name for the game instance.
 */

/**
 * Run the main game loop.
 *
 * This function blocks and drives the game's update/render loop until the game stops.
 */

/**
 * Push a new game state onto the state stack.
 *
 * The provided state takes ownership and becomes the current active state.
 * @param state Unique pointer to the game state to push.
 */

/**
 * Pop the current game state from the state stack.
 *
 * Removes the top-most state, allowing the previous state (if any) to resume.
 */
namespace raw::core {
class game {
private:
	raw::rendering::renderer							renderer;
	std::stack<std::unique_ptr<raw::core::game_state>> states;
	bool												is_running = true;

public:
	explicit game(const std::string& name);
	void run();

	void push_state(std::unique_ptr<raw::core::game_state> state);

	void pop_state();
};
} // namespace raw::core

#endif // SPACE_EXPLORER_GAME_H
