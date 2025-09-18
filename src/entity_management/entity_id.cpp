//
// Created by progamers on 9/18/25.
//
#pragma once
#include "entity_management/entity_id.h"
size_t std::hash<raw::entity_management::entity_id>::operator()(
	const raw::entity_management::entity_id& eid) const noexcept {
	size_t seed = std::hash<uint64_t> {}(eid.id);
	seed ^= std::hash<uint64_t> {}(eid.generation) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	return seed;
}