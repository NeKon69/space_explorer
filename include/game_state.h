//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_GAME_STATE_H
#define SPACE_EXPLORER_GAME_STATE_H

#include "clock.h"

namespace raw {
namespace rendering {
class renderer;
}
class game_state {
public:
	virtual ~game_state() = default;

	// It has bool return type so we can easily indicate if game was required to close
	virtual bool handle_input()							  = 0;
	virtual bool active()								  = 0;
	virtual void update(const raw::time& delta_time)	  = 0;
	virtual void draw(raw::rendering::renderer& renderer) = 0;
};
} // namespace raw

#endif // SPACE_EXPLORER_GAME_STATE_H
