//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_SIMULATION_STATE_H
#define SPACE_EXPLORER_SIMULATION_STATE_H

namespace raw {
    struct simulation_state {
        bool running;
        unsigned int amount_of_objects = 0;
        void add();
    };
}

#endif // SPACE_EXPLORER_SIMULATION_STATE_H
