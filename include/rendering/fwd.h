//
// Created by progamers on 8/26/25.
//

#pragma once
#include <vector>

namespace raw::rendering {
struct instance_data;
struct command;
class renderer;
struct visual_component;
using queue = std::vector<raw::rendering::command>;
} // namespace raw::rendering
