//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_CORE_FWD_H
#define SPACE_EXPLORER_CORE_FWD_H

namespace raw::core {
    class camera;
    class game;

    namespace camera_move {
        struct movement_state;
    }
} // namespace raw::core
namespace raw {
    class game_state;
    class shader;

    enum class time_rate { NANO, MICRO, MILLI, ORD };

    struct time;
    class clock;
} // namespace raw
#endif // SPACE_EXPLORER_CORE_FWD_H