//
// Created by progamers on 9/8/25.
//

#ifndef SPACE_EXPLORER_PENDING_ACTION_H
#define SPACE_EXPLORER_PENDING_ACTION_H
#include <cstdint>

#include "n_body/cuda/fwd.h"

namespace raw::n_body::cuda {
template<typename T>
struct pending_action {
	pending_action_type type;
	union {
		T		 object_to_add;
		uint32_t id_to_remove;
	};

	static pending_action add(T object_to_add) {
		pending_action action;
		action.type			 = pending_action_type::ADD;
		action.object_to_add = object_to_add;
		return action;
	}

	static pending_action remove(uint32_t id_to_remove) {
		pending_action action;
		action.type			= pending_action_type::REMOVE;
		action.id_to_remove = id_to_remove;
		return action;
	}
}
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_PENDING_ACTION_H
