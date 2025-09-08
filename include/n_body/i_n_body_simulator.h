//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_I_N_BODY_SIMULATOR_H
#define SPACE_EXPLORER_I_N_BODY_SIMULATOR_H
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
	graphics::graphics_data&					  graphics_data;
};
template<typename T>
class i_n_body_simulator {
protected:
	std::jthread			worker_thread;
	std::condition_variable condition;
	std::mutex				mutex;
	std::queue<task<T>>		task_queue;

public:
	virtual void step(core::time delta_time, std::shared_ptr<device_types::i_queue> queue,
					  std::shared_ptr<i_n_body_resource_manager<T>>, double g, double epsilon,
					  graphics::graphics_data& graphics_data) = 0;
	virtual ~i_n_body_simulator() {
	}
};
} // namespace raw::n_body

#endif // SPACE_EXPLORER_I_N_BODY_SIMULATOR_H
