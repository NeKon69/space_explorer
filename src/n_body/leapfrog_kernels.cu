//
// Created by progamers on 7/21/25.
//
#include <glm/gtc/matrix_transform.hpp>

#include "../../include/n_body/physics/leapfrog_kernels.h"
#include "../../include/n_body/physics/space_object.h"

template<typename T>
__device__ void compute_kick(raw::space_object<T> *objects, uint16_t count, uint16_t current, T g,
                             T epsilon, T dt) {
    auto a_total = glm::vec < 3, T
    >
    (0.0);
    // Using short here, since we don't plan on 5 billion planets to interact, maximum 10k
    for (uint16_t i = 0; i < count; ++i) {
        // Don't apply acceleration of itself
        if (current == i) {
            continue;
        }
        auto dist = objects[i].object_data.position - objects[current].object_data.position;
        T dist_sq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
        T inv_dist = rsqrt(dist_sq + epsilon * epsilon);
        T inv_dist_cubed = inv_dist * inv_dist * inv_dist;
        a_total += (g * objects[i].object_data.mass * inv_dist_cubed) * dist;
    }

    objects[current].object_data.velocity += a_total * dt;
}

template<typename T>
__global__ void compute_leapfrog(raw::space_object<T> *objects, glm::mat4 *objects_model,
                                 uint16_t count, T dt, T g) {
    const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= count) {
        return;
    }

    const auto epsilon = 1;

    // Kick
    compute_kick<T>(objects, count, x, g, epsilon, dt / 2);

    // Drift
    objects[x].object_data.position += objects[x].object_data.velocity * dt;

    // Works until 256 objects
    __syncthreads();

    // Kick
    compute_kick<T>(objects, count, x, g, epsilon, dt / 2);
    objects_model[x] = glm::mat4(1.0f);

    objects_model[x] = glm::scale(
        glm::translate(glm::mat4(1.0f), static_cast<glm::vec3>(objects[x].object_data.position)),
        glm::vec3(objects[x].object_data.radius));
}

template __global__ void compute_leapfrog<float>(raw::space_object<float> *objects, glm::mat4 *,
                                                 uint16_t count, float dt, float g);

template __global__ void compute_leapfrog<double>(raw::space_object<double> *, glm::mat4 *,
                                                  unsigned short, double, double);