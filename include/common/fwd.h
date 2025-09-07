//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_COMMON_FWD_H
#define SPACE_EXPLORER_COMMON_FWD_H
#include <chrono>

namespace raw {
struct vertex;
class mesh;
using std_clock = std::chrono::high_resolution_clock;
using UI		= unsigned int;
template<typename T>
using vec = std::vector<T>;
namespace common {
template<typename TResourceManager>
class scoped_resource_handle;
}
} // namespace raw
#endif // SPACE_EXPLORER_COMMON_FWD_H