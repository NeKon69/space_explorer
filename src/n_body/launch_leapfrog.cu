//
// Created by progamers on 7/21/25.
//
#include "n_body/launch_leapfrog.h"
#include "n_body/leapfrog_kernels.h"

namespace raw {
template<typename T>
void launch_leapfrog(raw::space_object<T>* objects_in, T time, uint16_t count, double g) {
	auto threads_per_block = 256;
	auto blocks			   = (count + threads_per_block - 1) / 256;
	if (count < 512) {
		threads_per_block = count % 32 == 0 ? count : (count / 32 + 1) * 32;
	}
	compute_leapfrog<T><<<blocks, threads_per_block>>>(objects_in, count, time, g);
}

template void launch_leapfrog<double>(space_object<double>* objects_in, double time, uint16_t count,
									  double g);
template void launch_leapfrog<float>(space_object<float>* objects_in, float time, uint16_t count,
									 double g);
} // namespace raw