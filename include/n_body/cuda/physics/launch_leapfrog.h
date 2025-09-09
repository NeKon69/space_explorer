//
// Created by progamers on 7/21/25.
//

#pragma once
#include <glm/glm.hpp>

#include "graphics/instanced_data.h"
#include "n_body/fwd.h"

namespace raw::n_body::cuda::physics {
template<typename T>
extern void launch_leapfrog(graphics::instanced_data* data, space_object_data<T>* objects,
							uint16_t count, double time, double g, double epsilon,
							cudaStream_t stream);

} // namespace raw::n_body::cuda::physics

