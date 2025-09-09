//
// Created by progamers on 7/24/25.
//

#pragma once

#include "common/fwd.h"
#include "rendering/fwd.h"

namespace raw::core {
class game_state {
public:
	virtual ~game_state() = default;

	// It has bool return type so we can easily indicate if game was required to close
	virtual bool handle_input() = 0;
	virtual bool active()		= 0;

	virtual void update(const raw::core::time& delta_time) = 0;
	virtual void draw(raw::rendering::renderer& renderer)  = 0;
};
} // namespace raw::core

