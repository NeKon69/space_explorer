//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_CUDA_N_BODY_SIMULATOR_H
#define SPACE_EXPLORER_CUDA_N_BODY_SIMULATOR_H
#include "device_types/device_ptr.h"
#include "n_body/i_n_body_resource_manager.h"
#include "n_body/i_n_body_simulator.h"
#include "physics/launch_leapfrog.h"
namespace raw::n_body::cuda {
template<typename T>
class n_body_simulator : public i_n_body_simulator<T> {
public:
	void step(core::time delta_time, std::shared_ptr<device_types::i_queue> queue,
			  std::shared_ptr<i_n_body_resource_manager<T>> source, double g, double epsilon,
			  graphics::graphics_data& graphics_data) override {
		sync();
		delta_time.to_sec();
		double sec			= delta_time.val;
		this->worker_thread = std::jthread([sec, queue, source, &graphics_data, g, epsilon]() {
			auto& cuda_q	   = dynamic_cast<device_types::cuda::cuda_stream&>(*queue);
			auto  local_stream = cuda_q.stream();
			graphics::gl_context_lock<graphics::context_type::N_BODY> lock(graphics_data);
			auto context	 = source->create_context();
			auto native_data = retrieve_data<device_types::backend::CUDA>(source->get_data());
			std::apply(n_body::cuda::physics::launch_leapfrog<T>, native_data, sec, g, epsilon,
					   local_stream);
		});
	}
};
} // namespace raw::n_body::cuda
#endif // SPACE_EXPLORER_CUDA_N_BODY_SIMULATOR_H
