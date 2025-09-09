//
// Created by progamers on 8/26/25.
//

#pragma once
#include <cmath>

#include "common/fwd.h"
#include "sphere_generation/fwd.h"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace raw::sphere_generation::cuda {

class sphere_resource_manager;
class sphere_generator;
// Stores 2 indices of vertices in the sphere

// Used by cuda since it accepts operators
struct edge : edge_base {
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
	edge()									 = default;
	HOST_DEVICE edge& operator=(edge&&)		 = default;
	HOST_DEVICE edge& operator=(const edge&) = default;
	HOST_DEVICE		  edge(edge&&)			 = default;
	HOST_DEVICE		  edge(const edge&)		 = default;
};
} // namespace raw::sphere_generation::cuda
