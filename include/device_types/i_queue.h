//
// Created by progamers on 9/7/25.
//

#pragma once
#include <CL/cl.h>
#include <memory>


#include "cuda/stream.h"
namespace raw::device_types {
struct i_bare_queue {
	virtual ~i_bare_queue() = default;
};

struct bare_stream : i_bare_queue {
	cudaStream_t stream = nullptr;
	explicit bare_stream(cudaStream_t s) : stream(s) {}
};

struct bare_queue : i_bare_queue {
	cl_command_queue queue = nullptr;
	explicit bare_queue(cl_command_queue q) : queue(q) {}
};

class i_queue {
public:

	virtual ~i_queue() = default;
	virtual std::unique_ptr<i_bare_queue> get_queue() const = 0;
	virtual void sync() = 0;
};
} // namespace raw::device_types

