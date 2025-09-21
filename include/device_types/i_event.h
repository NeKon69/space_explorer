//
// Created by progamers on 9/21/25.
//

#include <CL/cl.h>
#include <driver_types.h>

#include "device_types/i_queue.h"

namespace raw::device_types {
struct i_bare_event {
	virtual ~i_bare_event() = default;
};

struct cuda_bare_event : i_bare_event {
	cudaEvent_t event = nullptr;
	explicit cuda_bare_event(cudaEvent_t e) : event(e) {}
};

struct cl_bare_event : i_bare_queue {
	cl_event event = nullptr;
	explicit cl_bare_event(cl_event e) : event(e) {}
};
class i_event {
public:
	virtual ~i_event() = default;

	virtual void record(std::shared_ptr<i_queue>& queue) = 0;

	virtual void wait(std::shared_ptr<i_queue>& queue) = 0;

	virtual void sync() = 0;

	virtual bool is_ready() = 0;
};
} // namespace raw::device_types