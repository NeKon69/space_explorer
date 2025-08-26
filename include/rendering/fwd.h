//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_RENDERING_FWD_H
#define SPACE_EXPLORER_RENDERING_FWD_H
#include <vector>

namespace raw::rendering {
    struct instance_data;
    struct command;
    using queue = std::vector<raw::rendering::command>;
    class renderer;
    struct vertex;
} // namespace raw::rendering
#endif // SPACE_EXPLORER_RENDERING_FWD_H