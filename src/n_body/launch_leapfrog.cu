//
// Created by progamers on 7/21/25.
//
#include "n_body/launch_leapfrog.h"
#include "n_body/leapfrog_kernels.h"
#include "n_body/space_object.h"

namespace raw {
template void launch_leapfrog<double>(raw::space_object<double>* objects_in, double time,
									  uint16_t count, double g);
template void launch_leapfrog<float>(raw::space_object<float>*, float, unsigned short, double);
template<typename T>
void launch_leapfrog(raw::space_object<T>* objects_in, T time, uint16_t count, double g) {
	auto threads_per_block = 256;
	auto blocks			   = (count + threads_per_block - 1) / 256;
	if (count < 512) {
		threads_per_block = count % 32 == 0 ? count : (count / 32 + 1) * 32;
	}
	compute_leapfrog<T><<<blocks, threads_per_block>>>(objects_in, count, time, static_cast<T>(g));
}

} // namespace raw