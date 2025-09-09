//
// Created by progamers on 8/26/25.
//

#pragma once

namespace raw::core {
class game;

namespace camera {
struct movement_state;
class camera;
class player_controller;
} // namespace camera
} // namespace raw::core

namespace raw::core {
class game_state;

enum class time_rate { NANO, MICRO, MILLI, ORD };

struct time;
class clock;
} // namespace raw::core
