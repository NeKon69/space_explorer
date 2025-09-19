//
// Created by progamers on 7/21/25.
//
#include "device_types/cuda/error.h"
#include "n_body/cuda/physics/launch_leapfrog.h"
#include "n_body/cuda/physics/leapfrog_kernels.h"
#include "n_body/fwd.h"

namespace raw::n_body::cuda::physics {
template void launch_leapfrog<double>(graphics::instanced_data*	 data,
									  physics_component<double>* objects, uint16_t count,
									  double time, double g, double epsilon, cudaStream_t stream);

template void launch_leapfrog<float>(graphics::instanced_data* data,
									 physics_component<float>* objects, uint16_t count, double time,
									 double g, double epsilon, cudaStream_t stream);
template<typename T>
void launch_leapfrog(graphics::instanced_data* data, physics_component<T>* objects, uint16_t count,
					 double time, double g, double epsilon, cudaStream_t stream) {
	auto threads_per_block = 1024;
	auto blocks			   = (count + threads_per_block - 1) / 256;
	if (count < 1024) {
		threads_per_block = count % 32 == 0 ? count : (count / 32 + 1) * 32;
	}
	// this is stupid algorithm. probably one day in the future should do O(n log n)
	compute_k<T><<<blocks, threads_per_block, 0, stream>>>(data, objects, count, time, g, epsilon);
	compute_d<T><<<blocks, threads_per_block, 0, stream>>>(objects, count, time);
	compute_k_final<T>
		<<<blocks, threads_per_block, 0, stream>>>(data, objects, count, time, g, epsilon);
}
} // namespace raw::n_body::cuda::physics