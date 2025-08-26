//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_SPHERE_GENERATION_FWD_H
#define SPACE_EXPLORER_SPHERE_GENERATION_FWD_H
#include "common/fwd.h"

namespace raw {
    namespace predef {
        // Oh and btw, turns out, even after 4 steps our sphere gets nearly perfect (even on 2k monitor,
        // well maybe on 4k it would be nice to have 6, but 4 is pretty much enough)
        static constexpr auto BASIC_RADIUS = 1.0f;
        static constexpr auto BASIC_STEPS = 7U;
        static constexpr auto MAX_STEPS = 8U;
        // That you can't change, all things above you can
        static constexpr auto BASIC_AMOUNT_OF_TRIANGLES = 20U;
    } // namespace predef
    namespace predef {
        static const UI MAXIMUM_AMOUNT_OF_INDICES =
                BASIC_AMOUNT_OF_TRIANGLES * static_cast<UI>(std::pow(4, MAX_STEPS)) + 2;
        static const UI MAXIMUM_AMOUNT_OF_VERTICES = 10 * static_cast<UI>(std::pow(4, MAX_STEPS));
    } // namespace predef

    class icosahedron_generator;
} // namespace raw
#endif // SPACE_EXPLORER_SPHERE_GENERATION_FWD_H