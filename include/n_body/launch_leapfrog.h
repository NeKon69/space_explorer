//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#define SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#include "leapfrog_kernels.h"
namespace raw {
template<typename T>
class space_object;
template<typename T>
void launch_leapfrog(space_object<T>* objects_in, T time, uint16_t count, double g);
} // namespace raw

#endif // SPACE_EXPLORER_LAUNCH_LEAPFROG_H
