//
// Created by progamers on 9/7/25.
//

#pragma once
#include "common/to_raw_data.h"
#include "device_types/cuda/stream.h"
#include "device_types/device_ptr.h"
#include "n_body/i_n_body_resource_manager.h"
#include "n_body/i_n_body_simulator.h"
#include "physics/launch_leapfrog.h"
namespace raw::n_body::cuda {

template<typename T>
struct is_soa_device_data : std::false_type {};
template<typename T>
struct is_soa_device_data<soa_device_data<T>> : std::true_type {};

template<typename T>
constexpr bool is_soa_device_data_v = is_soa_device_data<std::decay_t<T>>::value;

template<typename>
struct get_soa_template_arg;
template<typename T>
struct get_soa_template_arg<soa_device_data<T>> {
	using type = T;
};

namespace detail {
template<auto B, typename T>
auto convert_element(T&& element) {
	if constexpr (is_soa_device_data_v<T>) {
		using InnerType = typename get_soa_template_arg<std::decay_t<T>>::type;
		return object_data_view<InnerType>(std::forward<T>(element));
	} else {
		return common::transform_element<B>(std::forward<T>(element));
	}
}
} // namespace detail

template<backend B, typename SourceTuple>
auto convert_data(SourceTuple&& source) {
	return std::apply(
		[]<typename... T>(const T&... elements) {
			std::make_tuple(detail::convert_element<B>(std::forward<T>(elements))...);
		},
		std::forward<SourceTuple>(source));
}

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
					// Acquire lock on first task
					lock =
						std::make_unique<graphics::gl_context_lock<graphics::context_type::N_BODY>>(
							*curr_task.graphics_data);
				}
				auto& cuda_q = dynamic_cast<device_types::cuda::cuda_stream&>(*curr_task.queue);
				{
					auto local_stream = cuda_q.stream();
					auto context	  = curr_task.manager->create_context();
					auto native_data  = convert_data<backend::CUDA>(curr_task.manager->get_data());
					std::apply(n_body::cuda::physics::launch_leapfrog<T>,
							   std::tuple_cat(native_data,
											  std::make_tuple(curr_task.delta_time, curr_task.g,
															  curr_task.epsilon, local_stream)));
				}
				cuda_q.sync();
				{
					std::lock_guard locker(this->mutex);
					--this->tasks_in_queue;
					if (this->tasks_in_queue == 0) {
						this->sync_condition.notify_one();
					}
				}
				++i;
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
