//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#define SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#include "space_object.h"
namespace raw {
void launch_leapfrog(raw::space_object* objects_in, time since_last_upd,
					 uint16_t count, double g);
}

#endif // SPACE_EXPLORER_LAUNCH_LEAPFROG_H
