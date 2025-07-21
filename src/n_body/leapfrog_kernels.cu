//
// Created by progamers on 7/21/25.
//
#include "n_body/leapfrog_kernels.h"

__device__ void compute_kick(raw::space_object* objects, uint16_t count, uint16_t current, double g,
							 double epsilon, double dt) {
	auto a_total = glm::dvec3(0.0);
	// Using short here, since we don't plan on 5 billion planets to interact, maximum 10k
	for (uint16_t i = 0; i < count; ++i) {
		// Don't apply acceleration of itself
		if (current == i) {
			continue;
		}
		auto   dist			  = objects[i].get().position - objects[current].get().position;
		double dist_sq		  = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
		double inv_dist		  = rsqrt(dist_sq + epsilon * epsilon);
		double inv_dist_cubed = inv_dist * inv_dist * inv_dist;
		a_total += (g * objects[i].get().mass * inv_dist_cubed) * dist;
	}

	objects[current].get().velocity += a_total * dt;
}

__global__ void compute_leapfrog(raw::space_object* objects, uint16_t count, double dt, double g) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	const auto epsilon = 0.01;

	// Kick
	compute_kick(objects, count, x, g, epsilon, dt);

	// Drift
	objects[x].get().position += objects[x].get().velocity * dt;

	__syncthreads();

	// Kick
	compute_kick(objects, count, x, g, epsilon, dt);
}

