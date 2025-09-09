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
}
} // namespace raw
