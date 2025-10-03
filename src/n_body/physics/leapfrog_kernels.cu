//
// Created by progamers on 7/21/25.
//

#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

#include "n_body/cuda/physics/leapfrog_kernels.h"
#include "n_body/physics_component.h"
namespace raw::n_body::cuda::physics {
template<typename T>
__device__ void print_matrix_device(const glm::mat4 &m, const char *label) {
	printf("--- %s (Device) ---\n", label);
	for (int i = 0; i < 4; ++i) {
		printf("[ %10.4f %10.4f %10.4f %10.4f ]\n", m[0][i], m[1][i], m[2][i], m[3][i]);
	}
	printf("-------------------------\n");
}

template<typename T>
__device__ void compute_kick_rw(const space_object_data<T> *read_ptr,
								space_object_data<T> *write_ptr, uint16_t count, uint16_t current,
								T g, T epsilon, T dt) {
	auto						a_total		   = glm::vec<3, T>(0.0);
	const space_object_data<T> &current_object = read_ptr[current];
	// Using short here, since we don't plan on 5 billion planets to interact, maximum 10k
	for (uint16_t i = 0; i < count; ++i) {
		// Don't apply acceleration of itself
		if (current == i) {
			continue;
		}
		const space_object_data<T> &local_object = read_ptr[i];
		glm::vec<3, T>				dist		 = local_object.position - current_object.position;
		T							dist_sq	 = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
		T							inv_dist = rsqrt(dist_sq + epsilon * epsilon);
		T							inv_dist_cubed = inv_dist * inv_dist * inv_dist;
		a_total += (g * local_object.mass * inv_dist_cubed) * dist;
	}

	write_ptr[current].velocity += a_total * dt;
	if (write_ptr != read_ptr) {
		write_ptr[current].position = read_ptr[current].position;
	}
}

template<typename T>
__device__ void compute_kick(const space_object_data<T> *object_old, space_object_data<T> *objects,
							 uint16_t count, uint16_t current, T g, T epsilon, T dt) {
	compute_kick_rw(object_old, objects, count, current, g, epsilon, dt);
}
template<typename T>
__device__ void compute_kick(space_object_data<T> *objects, uint16_t count, uint16_t current, T g,
							 T epsilon, T dt) {
	compute_kick_rw(objects, objects, count, current, g, epsilon, dt);
}

template<typename T>
__global__ void compute_k(graphics::instanced_data *data, const space_object_data<T> *object_old,
						  space_object_data<T> *objects, uint16_t count, T dt, T g, T epsilon) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	// Kick
	compute_kick<T>(object_old, objects, count, x, g, epsilon, dt / 2);
}
template<typename T>
__global__ void compute_d(space_object_data<T> *objects, uint16_t count, T dt) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	objects[x].position += dt * objects[x].velocity;
}

template<typename T>
__global__ void compute_k_final(graphics::instanced_data *data, space_object_data<T> *objects,
								uint16_t count, T dt, T g, T epsilon) {
	const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= count) {
		return;
	}

	compute_kick<T>(objects, count, x, g, epsilon, dt / 2);

	data[x].model = glm::mat4(1.0f);

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
template __global__ void compute_d<float>(space_object_data<float> *objects, uint16_t count,
										  float dt);
template __global__ void compute_d<double>(space_object_data<double> *objects, uint16_t count,
										   double dt);
template __global__ void compute_k_final<float>(graphics::instanced_data *data,
												space_object_data<float> *objects, uint16_t count,
												float dt, float g, float epsilon);
template __global__ void compute_k_final<double>(graphics::instanced_data  *data,
												 space_object_data<double> *objects, uint16_t count,
												 double dt, double g, double epsilon);

} // namespace raw::n_body::cuda::physics