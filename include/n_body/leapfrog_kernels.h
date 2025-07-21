//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LEAPFROG_KERNEL_H
#define SPACE_EXPLORER_LEAPFROG_KERNEL_H
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "space_object.h"
__device__ void compute_acceleration(raw::space_object* objects, uint16_t count, uint16_t current,
							 double g, double epsilon);

__device__ void update_position(raw::space_object* objects_in,
                                raw::space_object* objects_out, raw::time since_last_upd, uint16_t count);

__global__ void compute_leapfrog(raw::space_object* objects_in,
									 raw::space_object* objects_out, raw::time since_last_upd, uint16_t count, double g);


#endif // SPACE_EXPLORER_LEAPFROG_KERNEL_H
