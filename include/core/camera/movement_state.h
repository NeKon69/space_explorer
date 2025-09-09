//
// Created by progamers on 7/24/25.
//

#pragma once
namespace raw::core::camera {
struct movement_state {
	bool forward  = false;
	bool backward = false;
	bool left	  = false;
	bool right	  = false;
	bool up		  = false;
	bool down	  = false;
};
} // namespace raw::core::camera
