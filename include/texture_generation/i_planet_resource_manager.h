//
// Created by progamers on 9/12/25.
//

#pragma once
#include <tuple>

#include "common/scoped_resource_handle.h"
#include "texture_generation/fwd.h"

namespace raw::texture_generation {
// I don't know yet what i will need, but this will do for now
class i_planet_resource_manager {
protected:
	friend texture_generation_context;

public:
	virtual texture_generation_data	   get_data(planet_id id, LOD_LEVEL lod_level) = 0;
	virtual texture_generation_context create_context(planet_id id, LOD_LEVEL lod_level) = 0;

	virtual ~i_planet_resource_manager() = default;
};
} // namespace raw::texture_generation
