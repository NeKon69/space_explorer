//
// Created by progamers on 8/26/25.
//

#pragma once
#include <chrono>
#include <vector>

namespace raw {
struct vertex;
class mesh;
using std_clock = std::chrono::high_resolution_clock;
using UI		= unsigned int;
namespace common {
template<typename TResourceManager>
class scoped_resource_handle;
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
} // namespace common
} // namespace raw
