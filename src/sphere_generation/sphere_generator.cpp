//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/sphere_generator.h"

#include "sphere_generation/icosahedron_data_manager.h"
#include "sphere_generation/kernel_launcher.h"

namespace raw::sphere_generation {
void sphere_generator::generate(UI steps, cuda_types::cuda_stream& stream,
								icosahedron_data_manager& source) {
	if (steps >= predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	sync();
	launch_dummy_kernel();
	auto context		 = source.create_context();
	auto data_for_thread = source.get_data();
	// worker_thread		 = std::jthread([data = std::move(data_for_thread), &stream, steps,
	// &source] mutable {
	cudaStream_t local_stream = stream.stream();
	std::apply(
		sphere_generation::launch_tessellation,
		std::tuple_cat(std::move(data_for_thread), std::make_tuple(std::ref(local_stream), steps)));
	// });
}
void sphere_generator::sync() {
	if (worker_thread.joinable())
		worker_thread.join();
}

} // namespace raw::sphere_generation