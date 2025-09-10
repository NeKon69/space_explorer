//
// Created by progamers on 9/7/25.
//

#pragma once
#include <oneapi/tbb/detail/_task.h>

#include <condition_variable>
#include <queue>
#include <thread>

#include "core/clock.h"
#include "device_types/i_queue.h"
#include "graphics/gl_context_lock.h"
#include "n_body/fwd.h"
namespace raw::n_body {
template<typename T>
struct task {
	double										  delta_time;
	std::shared_ptr<device_types::i_queue>		  queue;
	std::shared_ptr<i_n_body_resource_manager<T>> manager;
	double										  g;
	double										  epsilon;
	graphics::graphics_data*					  graphics_data;
};
template<typename T>
class i_n_body_simulator {
protected:
	std::condition_variable condition;
	std::condition_variable sync_condition;
	std::jthread			worker_thread;
	std::mutex				mutex;
	std::queue<task<T>>		task_queue;
	size_t					tasks_in_queue = 0;

public:
	virtual void step(core::time delta_time, std::shared_ptr<device_types::i_queue> queue,
					  std::shared_ptr<i_n_body_resource_manager<T>> source, double g,
					  double epsilon, graphics::graphics_data& graphics_data) = 0;
	void		 sync() {
		std::unique_lock locker(this->mutex);
		this->sync_condition.wait(locker, [this] {
			return this->tasks_in_queue == 0;
		});
	}
	virtual ~i_n_body_simulator() {
		worker_thread.request_stop();
		condition.notify_one();
		if (worker_thread.joinable()) {
			worker_thread.join();
		}
	}
};
} // namespace raw::n_body
