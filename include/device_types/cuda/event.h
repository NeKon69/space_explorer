//
// Created by progamers on 9/21/25.
//

#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "device_types/cuda/fwd.h"
#include "device_types/i_event.h"
#include "error.h"
namespace raw::device_types::cuda {
class event : public i_event {
private:
	cudaEvent_t event_;

public:
	event();
	~event() override;

	void record(std::shared_ptr<i_queue>& queue) override;
	void wait(std::shared_ptr<i_queue>& queue) override;
	void sync() override;
	bool is_ready() override;
};
} // namespace raw::device_types::cuda