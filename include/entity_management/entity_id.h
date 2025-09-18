//
// Created by progamers on 9/18/25.
//

#pragma once
#include <cstdint>
#include <queue>
#include <vector>

#include "entity_management/fwd.h"
namespace raw::entity_management {
struct entity_id {
	uint64_t id			= 0;
	uint64_t generation = 0;

	std::strong_ordering operator<=>(const entity_id&) const = default;
};
} // namespace raw::entity_management
template<>
struct std::hash<raw::entity_management::entity_id> {
	size_t operator()(const raw::entity_management::entity_id& eid) const noexcept;
};