//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#define SPACE_EXPLORER_LAUNCH_LEAPFROG_H
#include "n_body/fwd.h"
#include <glm/glm.hpp>

namespace raw {
    template<typename T>
    void launch_leapfrog(space_object<T> *objects, glm::mat4 *objects_model, T time, uint16_t count,
                         double g, cudaStream_t stream);
} // namespace raw

#endif // SPACE_EXPLORER_LAUNCH_LEAPFROG_H