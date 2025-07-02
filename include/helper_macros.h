//
// Created by progamers on 6/27/25.
//

#ifndef SPACE_EXPLORER_HELPER_MACROS_H
#define SPACE_EXPLORER_HELPER_MACROS_H
#include <chrono>
#include <vector>

namespace raw {
#define PASSIVE_VALUE static constexpr auto
using std_clock = std::chrono::high_resolution_clock;
using UI		= unsigned int;
template<typename T>
using vec = std::vector<T>;
} // namespace raw

#endif // SPACE_EXPLORER_HELPER_MACROS_H
