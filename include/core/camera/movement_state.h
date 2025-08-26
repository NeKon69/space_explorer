//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_MOVEMENT_STATE_H
#define SPACE_EXPLORER_MOVEMENT_STATE_H
namespace raw::core::camera_move {
	struct movement_state {
		bool forward = false;
		bool backward = false;
	bool left	  = false;
	bool right	  = false;
	bool up		  = false;
	bool down	  = false;
};
} // namespace raw
#endif // SPACE_EXPLORER_MOVEMENT_STATE_H
