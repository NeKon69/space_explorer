//
// Created by progamers on 7/21/25.
//
#include <cuda/std/__type_traits/is_primary_template.h>

#include <glm/gtc/matrix_transform.hpp>

#include "n_body/cuda/physics/leapfrog_kernels.h"
#include "n_body/cuda/physics/space_object.h"

namespace raw::n_body::cuda::physics {

template<typename T>
__device__ void compute_kick(space_object<T> *objects, uint16_t count, uint16_t current, T g,
							 T epsilon, T dt) {
	auto				 a_total		= glm::vec<3, T>(0.0);
	space_object_data<T> current_object = objects[current];
	// Using short here, since we don't plan on 5 billion planets to interact, maximum 10k
	for (uint16_t i = 0; i < count; ++i) {
		// Don't apply acceleration of itself
		if (current == i) {
			continue;
		}
		space_object_data<T> local_object = objects[i];
		auto dist	  = local_object.object_data.position - current_object.object_data.position;
		T	 dist_sq  = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
		T	 inv_dist = rsqrt(dist_sq + epsilon * epsilon);
		T	 inv_dist_cubed = inv_dist * inv_dist * inv_dist;
		a_total += (g * local_object.object_data.mass * inv_dist_cubed) * dist;
	}

	objects[current].object_data.velocity += a_total * dt;
}

template<typename T>
__global__ void compute_kd(space_object_data<T> *objects, uint16_t count, T dt, T g, T epsilon) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	// Kick
	compute_kick<T>(objects, count, x, g, epsilon, dt / 2);

	// Drift
	objects[x].object_data.position += objects[x].object_data.velocity * dt;
}

template<typename T>
__global__ void compute_k(graphics::instanced_data *data, space_object_data<T> *objects,
						  uint16_t count, T dt, T g, T epsilon) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	compute_kick<T>(objects, count, x, g, epsilon, dt);

	objects[x] = glm::mat4(1.0f);

	data[x].model =
		glm::scale(glm::translate(glm::mat4(1.0f), static_cast<glm::vec3>(objects[x].position)),
				   glm::vec3(objects[x].radius));
}

template __global__ void compute_k<float>(graphics::instanced_data *data,
										  space_object_data<float> *objects, uint16_t count,
										  float dt, float g, float epsilon);

template __global__ void compute_k<double>(graphics::instanced_data	 *data,
										   space_object_data<double> *objects, uint16_t count,
										   double dt, double g, double epsilon);
template __global__ void compute_kd<float>(space_object_data<float> *objects, uint16_t count,
										   float dt, float g, float epsilon);
template __global__ void compute_kd<double>(space_object_data<double> *objects, uint16_t count,
											double dt, double g, double epsilon);

} // namespace raw::n_body::cuda::physics