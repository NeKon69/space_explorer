//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/sphere_generator.h"

#include "sphere_generation/icosahedron_data_manager.h"
#include "sphere_generation/kernel_launcher.h"

namespace raw::sphere_generation {
/**
 * @brief Start an asynchronous sphere tessellation task.
 *
 * Validates the requested tessellation steps, waits for any existing worker to finish,
 * then spawns a background thread that launches tessellation using the provided
 * CUDA stream, icosahedron source data, and graphics context.
 *
 * The background worker stores its thread in the member `worker_thread`, acquires the
 * GL tessellation context, and invokes the tessellation launcher on the provided CUDA stream.
 *
 * @param steps Number of tessellation subdivision steps; must be less than predef::MAX_STEPS.
 *
 * @throws std::runtime_error if `steps` is greater than or equal to predef::MAX_STEPS.
 */
void sphere_generator::generate(UI steps, cuda_types::cuda_stream& stream,
								icosahedron_data_manager& source,
								graphics::graphics_data&  graphics_data) {
	if (steps >= predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	sync();
	auto data_for_thread = source.get_data();
	worker_thread		 = std::jthread([data = std::move(data_for_thread), &stream, steps, &source,
									 &graphics_data] mutable {
		   cudaStream_t local_stream = stream.stream();
		   graphics::gl_context_lock<graphics::context_type::TESS> lock(graphics_data);
		   auto context = source.create_context();
		   std::apply(sphere_generation::launch_tessellation,
						  std::tuple_cat(std::move(data), std::make_tuple(std::ref(local_stream), steps)));
	   });
}
void sphere_generator::sync() {
	if (worker_thread.joinable())
		worker_thread.join();
}

} // namespace raw::sphere_generation