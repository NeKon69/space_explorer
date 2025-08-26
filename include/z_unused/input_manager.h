//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_INPUT_MANAGER_H
#define SPACE_EXPLORER_INPUT_MANAGER_H

#include <functional>

#include "SDL3/SDL.h"
#include "../core/clock.h"
#include "z_unused/command.h"
#include <raw_memory.h>

namespace raw {
    class input_manager {
    private:
        std::unordered_map<SDL_Scancode, std::shared_ptr<raw::command::command> > key_bindings;
        std::vector<std::shared_ptr<raw::command::command> > command_queue;

    public:
        input_manager() = default;

        void init();

        void handle_events();

        std::vector<std::shared_ptr<raw::command::command> > &&get_commands();
    };
} // namespace raw

#endif // SPACE_EXPLORER_INPUT_MANAGER_H