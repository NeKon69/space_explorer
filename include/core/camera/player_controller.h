//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_PLAYER_CONTROLLER_H
#define SPACE_EXPLORER_PLAYER_CONTROLLER_H
#include <glm/glm.hpp>

#include "core/fwd.h"
#include "../../cuda_types/error.h"

namespace raw {
    namespace predef {
        static constexpr auto ACCELERATION = 20.0f;
        static constexpr auto MAX_SPEED = 10.0f;
        static constexpr auto FRICTION = 30.0f;
    } // namespace predef

    class player_controller {
    private:
        raw::core::camera &camera;
        glm::vec3 velocity;

    public:
        explicit player_controller(raw::core::camera &cam);

        void update(const raw::core::camera_move::movement_state &state, float delta_time);
    };
} // namespace raw
#endif // SPACE_EXPLORER_PLAYER_CONTROLLER_H