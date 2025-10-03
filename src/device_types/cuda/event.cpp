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
	i_queue* base_ptr	= queue.get();
	auto	 stream_ptr = dynamic_cast<cuda_stream*>(base_ptr);
	if (stream_ptr) {
		CUDA_SAFE_CALL(cudaEventRecord(event_, stream_ptr->stream()));
	} else {
		throw std::runtime_error("Passed the wrong queue type!");
	}
}
void event::wait(std::shared_ptr<i_queue>& queue) {
	i_queue* base_ptr	= queue.get();
	auto	 stream_ptr = dynamic_cast<cuda_stream*>(base_ptr);
	if (stream_ptr) {
		CUDA_SAFE_CALL(cudaStreamWaitEvent(stream_ptr->stream(), event_));
	} else {
		throw std::runtime_error("Passed the wrong queue type!");
	}
}
void event::sync() {
	CUDA_SAFE_CALL(cudaEventSynchronize(event_));
}

bool event::is_ready() {
	if (cudaError_t err = cudaEventQuery(event_); err == cudaSuccess) {
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