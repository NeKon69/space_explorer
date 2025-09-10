//
// Created by progamers on 9/7/25.
//

#pragma once
#include "common/to_raw_data.h"
#include "device_types/device_ptr.h"
#include "n_body/i_n_body_resource_manager.h"
#include "n_body/i_n_body_simulator.h"
#include "physics/launch_leapfrog.h"
namespace raw::n_body::cuda {
template<typename T>
class n_body_simulator : public i_n_body_simulator<T> {
public:
	n_body_simulator() {
		this->worker_thread = std::jthread([this](std::stop_token token) {
			uint32_t																   i = 0;
			std::unique_ptr<graphics::gl_context_lock<graphics::context_type::N_BODY>> lock;

			while (!token.stop_requested()) {
				task<T> curr_task;
				{
					std::unique_lock locker(this->mutex);
					this->condition.wait(locker, [this, &token] {
						return !this->task_queue.empty() || token.stop_requested();
					});
					if (token.stop_requested() && this->task_queue.empty()) {
						break;
					}
					curr_task = this->task_queue.front();
					this->task_queue.pop();
				}
				if (i == 0) {
					lock =
						std::make_unique<graphics::gl_context_lock<graphics::context_type::N_BODY>>(
							*curr_task.graphics_data);
				}
				auto& cuda_q = dynamic_cast<device_types::cuda::cuda_stream&>(*curr_task.queue);
				auto  local_stream = cuda_q.stream();
				{
					auto context	 = curr_task.manager->create_context();
					auto native_data = common::retrieve_data<device_types::backend::CUDA>(
						curr_task.manager->get_data());
					std::apply(n_body::cuda::physics::launch_leapfrog<T>,
							   std::tuple_cat(native_data,
											  std::make_tuple(curr_task.delta_time, curr_task.g,
															  curr_task.epsilon, local_stream)));
				}
				cuda_q.sync();
				{
					std::lock_guard lock(this->mutex);
					--this->tasks_in_queue;
					if (this->tasks_in_queue == 0) {
						this->sync_condition.notify_one();
					}
				}
				++i;
				std::cout << "I FINISHED TASK WITH N-BODY!!!!!!!!!\n";
			}
		});
	}
	void step(core::time delta_time, std::shared_ptr<device_types::i_queue> queue,
			  std::shared_ptr<i_n_body_resource_manager<T>> source, double g, double epsilon,
			  graphics::graphics_data& graphics_data) override {
		delta_time.to_sec();
		{
			double			sec = delta_time.val;
			std::lock_guard lock(this->mutex);
			this->task_queue.emplace(sec, queue, source, g, epsilon, std::addressof(graphics_data));
			++this->tasks_in_queue;
		}
		this->condition.notify_one();
	}
};
} // namespace raw::n_body::cuda
