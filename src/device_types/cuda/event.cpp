//
// Created by progamers on 9/21/25.
//
#include "device_types/cuda/event.h"

#include <iostream>

#include "device_types/cuda/stream.h"
namespace raw::device_types::cuda {
event::event() {
	CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}
void event::record(std::shared_ptr<i_queue>& queue) {
	auto& stream = std::dynamic_pointer_cast<cuda_stream>(queue);
	CUDA_SAFE_CALL(cudaEventRecord(event_, stream->stream()));
}
void event::wait(std::shared_ptr<i_queue>& queue) {
	auto& stream = std::dynamic_pointer_cast<cuda_stream>(queue);
	CUDA_SAFE_CALL(cudaStreamWaitEvent(stream->stream(), event_));
}
void event::sync() {
	CUDA_SAFE_CALL(cudaEventSynchronize(event_));
}

bool event::is_ready() {
	cudaError_t err = cudaEventQuery(event_);
	if (err == cudaSuccess) {
		return true;
	} else if (err == cudaErrorNotReady) {
		return false;
	} else {
		CUDA_FAIL_ON_ERROR(err);
	}
	return false;
}

event::~event() {
	try {
		CUDA_SAFE_CALL(cudaEventDestroy(event_));
	} catch (const cuda_exception& e) {
		std::cerr << e.what() << "\n";
	}
}

} // namespace raw::device_types::cuda