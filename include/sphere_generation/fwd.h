//
// Created by progamers on 9/4/25.
//

#pragma once
#include <cstdint>

#include "common/fwd.h"
namespace raw::sphere_generation {
namespace predef {
// Oh and btw, turns out, even after 4 steps our sphere gets nearly perfect (even on 2k monitor,
// well maybe on 4k it would be nice to have 6, but 4 is pretty much enough)
static constexpr auto BASIC_RADIUS = 1.0f;
static constexpr auto BASIC_STEPS  = 4U;
static constexpr auto MAX_STEPS	   = 8U;
// That you can't change, all things above you can
static constexpr uint32_t BASIC_AMOUNT_OF_TRIANGLES = 20U;
static constexpr uint32_t BASIC_AMOUNT_OF_VERTICES	= 12U;
static constexpr uint32_t BASIC_AMOUNT_OF_INDICES	= 60U;
static constexpr uint32_t MAXIMUM_AMOUNT_OF_INDICES =
	BASIC_AMOUNT_OF_TRIANGLES * (1u << (2u * MAX_STEPS)) * 3u;
static constexpr uint32_t MAXIMUM_AMOUNT_OF_VERTICES = 10u * (1u << (2u * MAX_STEPS)) + 2u;
static constexpr uint32_t MAXIMUM_AMOUNT_OF_TRIANGLES =
	BASIC_AMOUNT_OF_TRIANGLES * (1u << (2u * MAX_STEPS));
} // namespace predef

class i_sphere_resource_manager;
struct edge_base {
	uint32_t v0 = 0;
	uint32_t v1 = 0;
};
using generation_context = common::scoped_resource_handle<i_sphere_resource_manager>;
} // namespace raw::sphere_generation
