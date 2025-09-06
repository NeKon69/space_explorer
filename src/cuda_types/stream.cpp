//
// Created by progamers on 7/20/25.
//
#include "device_types/cuda/stream.h"

#include <iostream>
#include <memory>

#include "device_types/cuda/error.h"

namespace raw::device_types::cuda {
cuda_stream::cuda_stream() : created(std::make_shared<bool>(false)) {
	CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
	created = true;
}

void cuda_stream::destroy() {
	if (created)
		CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
	created = false;
}

void cuda_stream::destroy_noexcept() noexcept {
	try {
		destroy();
	} catch (const cuda_exception &e) {
		std::cerr << std::format("[CRITICAL] Destroying CUDA stream failed. \n{}", e.what());
	}
}

void cuda_stream::create() {
	destroy();
	if (!created)
		CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
	created = true;
}

void cuda_stream::sync() {
	CUDA_SAFE_CALL(cudaStreamSynchronize(_stream));
}

cuda_stream::cuda_stream(cuda_stream &&rhs) noexcept : _stream(rhs._stream), created(rhs.created) {
	rhs._stream = nullptr;
	rhs.created = false;
}

cuda_stream &cuda_stream::operator=(cuda_stream &&rhs) noexcept {
	destroy_noexcept();
	_stream		= rhs._stream;
	created		= rhs.created;
	rhs._stream = nullptr;
	rhs.created = false;
	return *this;
}

cudaStream_t &cuda_stream::stream() {
	return _stream;
}

cuda_stream::~cuda_stream() {
	destroy_noexcept();
}
} // namespace raw::device_types::cuda