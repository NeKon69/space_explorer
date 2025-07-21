//
// Created by progamers on 7/21/25.
//
#include "n_body/launch_leapfrog.h"
#include "n_body/leapfrog_kernels.h"

namespace raw {
void launch_leapfrog(raw::space_object* objects_in, time since_last_upd, uint16_t count, double g) {
	since_last_upd.to_milli();
	auto threads_per_block = 256;
	auto blocks			   = (count + threads_per_block - 1) / 256;
	if (count < 512) {
		threads_per_block = count % 32 == 0 ? count : (count / 32 + 1) * 32;
	}
	compute_leapfrog<<<blocks, threads_per_block>>>(objects_in, count,
													static_cast<double>(since_last_upd.val), g);
}
} // namespace raw