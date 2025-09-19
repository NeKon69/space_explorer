//
// Created by progamers on 9/18/25.
//
#include "entity_management/entity_manager.h"

#include <nvtx3/nvtx3.hpp>

#include "entity_management/entity_id.h"
namespace raw::entity_management {
template<typename T>
entity_id entity_manager<T>::create_entity() {
	uint64_t new_index = 0;
	if (free_indices.empty()) {
		new_index = generations.size();
		generations.push_back(0);
	} else {
		new_index = free_indices.front();
		free_indices.pop();
	}

	return entity_id {new_index, generations[new_index]};
}

template<typename T>
void entity_manager<T>::destroy_entity(entity_id entity) {
	if (!is_valid(entity)) {
		return;
	}

	// means it's new generation
	generations[entity.generation]++;
	free_indices.push(entity.id);
}

template<typename T>
bool entity_manager<T>::is_valid(entity_id entity) const {
	if (entity.id < generations.size()) {
		return entity.generation == generations[entity.id];
	}
	return false;
}

} // namespace raw::entity_management