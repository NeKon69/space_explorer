//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_DOWN_H
#define SPACE_EXPLORER_DOWN_H
#include "command.h"
namespace raw::command {
    class move_camera_down : public raw::command::command {
    public:
        void execute(raw::scene& scene) override;
    };
} // namespace raw::command
#endif // SPACE_EXPLORER_DOWN_H
