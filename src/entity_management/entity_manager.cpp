//
// Created by progamers on 9/18/25.
//
#include "entity_management/entity_manager.h"

#include <nvtx3/nvtx3.hpp>

#include "entity_management/entity_id.h"
namespace raw::entity_management {
entity_id entity_manager::create_entity() {
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
void entity_manager::destroy_entity(entity_id entity) {
	if (!is_valid(entity)) {
		return;
	}

	// means it's new generation
	generations[entity.generation]++;
	free_indices.push(entity.id);
}

bool entity_manager::is_valid(entity_id entity) const {
	if (entity.id < generations.size()) {
		return entity.generation == generations[entity.id];
	}
	return false;
}

} // namespace raw::entity_management