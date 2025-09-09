//
// Created by progamers on 7/24/25.
//

#pragma once

#include <memory>
#include <stack>

#include "core/fwd.h"
#include "game_state.h"
#include "rendering/renderer.h"

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

