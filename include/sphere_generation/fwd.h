//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_SPHERE_GENERATION_FWD_H
#define SPACE_EXPLORER_SPHERE_GENERATION_FWD_H
#include <cmath>

#include "common/fwd.h"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace raw::sphere_generation {
namespace predef {
// Oh and btw, turns out, even after 4 steps our sphere gets nearly perfect (even on 2k monitor,
// well maybe on 4k it would be nice to have 6, but 4 is pretty much enough)
static constexpr auto BASIC_RADIUS = 1.0f;
static constexpr auto BASIC_STEPS  = 7U;
static constexpr auto MAX_STEPS	   = 8U;
// That you can't change, all things above you can
static constexpr auto BASIC_AMOUNT_OF_TRIANGLES = 20U;
} // namespace predef
namespace predef {
static constexpr UI MAXIMUM_AMOUNT_OF_INDICES =
	BASIC_AMOUNT_OF_TRIANGLES * (1u << (2u * MAX_STEPS)) * 3u;
static constexpr UI MAXIMUM_AMOUNT_OF_VERTICES = 10u * (1u << (2u * MAX_STEPS)) + 2u;
static constexpr UI MAXIMUM_AMOUNT_OF_TRIANGLES =
	BASIC_AMOUNT_OF_TRIANGLES * (1u << (2u * MAX_STEPS));
} // namespace predef

class icosahedron_data_manager;
class sphere_generator;
class generation_context;
// Stores 2 indices of vertices in the sphere
struct edge {
	uint32_t		 v0;
	uint32_t		 v1;
	HOST_DEVICE bool operator<(const edge& other) const {
		if (v0 < other.v0) {
			return true;
		}
		if (v0 > other.v0) {
			return false;
		}
		return v1 < other.v1;
	}
	HOST_DEVICE bool operator==(const edge& edge) const {
		return v0 == edge.v0 && v1 == edge.v1;
	}
	HOST_DEVICE bool operator!=(const edge& edge) const {
		return !operator==(edge);
	}
};
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_SPHERE_GENERATION_FWD_H