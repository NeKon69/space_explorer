//
// Created by progamers on 7/21/25.
//
#include "n_body/leapfrog_kernels.h"
template<typename T>
__device__ void compute_kick(raw::space_object<T>* objects, uint16_t count, uint16_t current,
							 double g, double epsilon, double dt) {
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
template<typename T>
__global__ void compute_leapfrog(raw::space_object<T>* objects, uint16_t count, double dt,
								 double g) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	const auto epsilon = 1;

	// Kick
	compute_kick(objects, count, x, g, epsilon, dt);

	// Drift
	objects[x].get().position += objects[x].get().velocity * dt;

	// Works until 256 objects
	__syncthreads();

	// Kick
	compute_kick(objects, count, x, g, epsilon, dt);
}

template __global__ void compute_leapfrog<double>(raw::space_object<double>* objects,
												  uint16_t count, double dt, double g);
template __global__ void compute_leapfrog<float>(raw::space_object<float>* objects, uint16_t count,
												 float dt, float g);
template __device__ void compute_kick<double>(raw::space_object<double>* objects, uint16_t count,
											  uint16_t current, double g, double epsilon,
											  double dt);
template __device__ void compute_kick<float>(raw::space_object<float>* objects, uint16_t count,
											 uint16_t current, float g, float epsilon, float dt);
