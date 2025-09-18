//
// Created by progamers on 9/18/25.
//

#pragma once

#include <cstdint>
#include <queue>
#include <vector>

#include "entity_management/fwd.h"
namespace raw::entity_management {
class entity_manager {
private:
	std::vector<uint64_t> generations;
	std::queue<uint64_t>  free_indices;

public:
	entity_id create_entity();
	void	  destroy_entity(entity_id id);
	bool	  is_valid(entity_id id) const;
};
} // namespace raw::entity_management