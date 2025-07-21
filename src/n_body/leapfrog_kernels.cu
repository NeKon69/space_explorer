//
// Created by progamers on 7/21/25.
//
#include "n_body/leapfrog_kernels.h"

__device__ void compute_acceleration(raw::space_object* objects, uint16_t count,
									 uint16_t current, double g, double epsilon) {
	auto f_total = glm::dvec3(0.0);
	// Using short here, since we don't plan on 5 billion planets to interact, maximum 10k
	for (uint16_t i = 0; i < count; ++i) {
		// Don't apply acceleration of itself
		if (current == i) {
			return;
		}
		auto   dist			  = objects[i].get().position - objects[current].get().position;
		double dist_sq		  = pow(dist.x, 2) * pow(dist.y, 2) * pow(dist.z, 2);
		double inv_dist		  = rsqrt(dist_sq + pow(epsilon, 2));
		double inv_dist_cubed = pow(inv_dist, 3);
		double f_scalar		  = g * objects[i].get().mass * inv_dist_cubed;
		f_total += f_scalar * dist;
	}
	objects[current].get().acceleration = f_total / objects[current].get().mass;
}

__device__ void update_position(raw::space_object* objects_in,
                                raw::space_object* objects_out, raw::time since_last_upd,
                                uint16_t count) {
    const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= count) {
        return;
    }

    objects_out[x].get().velocity = objects_in[x].get().velocity +
                              objects_in[x].get().acceleration * static_cast<double>(since_last_upd.val);
    objects_out[x].get().position =
            objects_in[x].get().position + objects_out[x].get().velocity * static_cast<double>(since_last_upd.val);
}

__global__ void compute_leapfrog(raw::space_object* objects_in,
								 raw::space_object* objects_out, raw::time since_last_upd,
								 uint16_t count, double g) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}
	const auto epsilon = 0.01;
	compute_acceleration(objects_in, count, x, g, epsilon);
	update_position(objects_in, objects_out, since_last_upd, count);
}
