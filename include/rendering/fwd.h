//
// Created by progamers on 8/26/25.
//

#pragma once
#include <vector>

namespace raw::rendering {
struct instance_data;
struct command;
using queue = std::vector<raw::rendering::command>;
class renderer;
} // namespace raw::rendering
