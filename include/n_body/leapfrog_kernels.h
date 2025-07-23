//
// Created by progamers on 7/21/25.
//

#ifndef SPACE_EXPLORER_LEAPFROG_KERNEL_H
#define SPACE_EXPLORER_LEAPFROG_KERNEL_H
#include <cuda_runtime.h>

#include <glm/glm.hpp>

namespace raw {
template<typename T = double>
class space_object;
}
template<typename T = double>
extern __device__ void compute_kick(raw::space_object<T>* objects, uint16_t count, uint16_t current,
									T g, T epsilon, T dt);
template<typename T = double>
extern __global__ void compute_leapfrog(raw::space_object<T>* objects, glm::mat4* objects_model, uint16_t count, T dt,
										T g);


#endif // SPACE_EXPLORER_LEAPFROG_KERNEL_H
