//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_RENDER_COMMAND_H
#define SPACE_EXPLORER_RENDER_COMMAND_H
#include <raw_memory.h>
#include "rendering/fwd.h"
#include "mesh.h"
#include "core/shader/shader.h"

namespace raw::rendering {
    struct instance_data {
        UI vbo;
        int count;
    };

    struct command {
        raw::shared_ptr<raw::mesh> mesh;
        raw::shared_ptr<raw::shader> shader;
        std::optional<raw::rendering::instance_data> inst_data;
    };
} // namespace raw::rendering

#endif // SPACE_EXPLORER_RENDER_COMMAND_H