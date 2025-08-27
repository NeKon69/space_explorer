//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LEAPFROG_KERNEL_H
#define SPACE_EXPLORER_LEAPFROG_KERNEL_H
#include <glm/glm.hpp>

#include "n_body/fwd.h"

namespace raw::n_body::physics {
    template<typename T = double>
    extern __device__ void compute_kick(raw::n_body::physics::space_object<T> *objects, uint16_t count,
                                        uint16_t current, T g, T epsilon, T dt);

    template<typename T = double>
    extern __global__ void compute_leapfrog(raw::n_body::physics::space_object<T> *objects,
                                            glm::mat4 *objects_model, uint16_t count, T dt, T g);
} // namespace raw::n_body::physics
#endif // SPACE_EXPLORER_LEAPFROG_KERNEL_H