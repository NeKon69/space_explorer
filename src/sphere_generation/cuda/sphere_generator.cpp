//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/cuda/sphere_generator.h"

#include "common/to_raw_data.h"
#include "core/clock.h"
#include "device_types/device_ptr.h"
#include "sphere_generation/cuda/kernel_launcher.h"
#include "sphere_generation/cuda/sphere_resource_manager.h"

namespace raw::sphere_generation::cuda {

void sphere_generator::generate(uint32_t steps, std::shared_ptr<i_queue> stream,
								std::shared_ptr<i_sphere_resource_manager> source,
								graphics::graphics_data&				   graphics_data) {
	if (steps > predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	sync();
	worker_thread = std::jthread([&stream, steps, source, &graphics_data] mutable {
		core::clock clock;
		auto&		cuda_q		 = dynamic_cast<cuda_stream&>(*stream);
		auto		local_stream = cuda_q.stream();
		graphics::gl_context_lock<graphics::context_type::TESS> lock(graphics_data);
		auto													context = source->create_context();
		auto native_data = common::retrieve_data<device_types::backend::CUDA>(source->get_data());

		auto time = clock.restart();
		time.to_milli();
		std::cout << "Preparation took: " << time << std::endl;
		clock.restart();
		std::apply(
			launch_tessellation,
			std::tuple_cat(std::move(native_data), std::make_tuple(std::ref(local_stream), steps)));
		cuda_q.sync();
		time = clock.restart();
		time.to_milli();
		std::cout << "Tessellation with " << steps << " steps took: " << time << std::endl;
	});
}

} // namespace raw::sphere_generation::cuda