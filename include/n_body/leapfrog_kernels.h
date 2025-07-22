//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LEAPFROG_KERNEL_H
#define SPACE_EXPLORER_LEAPFROG_KERNEL_H
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "space_object.h"

template<typename T = double>
extern __device__ void compute_kick(raw::space_object<T>* objects, uint16_t count, uint16_t current,
									double g, double epsilon, double dt);
template<typename T = double>
extern __global__ void compute_leapfrog(raw::space_object<T>* objects, uint16_t count, double dt,
										double g);



#endif // SPACE_EXPLORER_LEAPFROG_KERNEL_H
