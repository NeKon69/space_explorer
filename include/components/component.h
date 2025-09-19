//
// Created by progamers on 9/19/25.
//
#pragma once

#include "common/fwd.h"
namespace raw::components {
struct component {};
template<typename T>
concept IsComponent = std::is_base_of_v<component, T>;
} // namespace raw::components