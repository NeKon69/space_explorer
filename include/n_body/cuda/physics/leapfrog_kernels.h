//
// Created by progamers on 7/21/25.
//

#pragma once
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "graphics/instanced_data.h"
#include "n_body/fwd.h"

namespace raw::n_body::cuda::physics {
template<typename T = double>
extern __device__ void compute_kick(space_object_data<T> *objects, uint16_t count, uint16_t current,
									T g, T epsilon, T dt);

template<typename T>
extern __global__ void compute_k(graphics::instanced_data *data, space_object_data<T> *objects,
								 uint16_t count, T dt, T g, T epsilon);

template<typename T>
extern __global__ void compute_d(space_object_data<T> *objects, uint16_t count, T dt);

template<typename T>
extern __global__ void compute_k_final(graphics::instanced_data *data,
									   space_object_data<T> *objects, uint16_t count, T dt, T g,
									   T epsilon);

} // namespace raw::n_body::cuda::physics
