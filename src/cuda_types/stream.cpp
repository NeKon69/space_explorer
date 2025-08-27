//
// Created by progamers on 7/20/25.
//
#include "cuda_types/stream.h"

#include <raw_memory.h>

#include "cuda_types/error.h"

namespace raw::cuda_types {
	cuda_stream::cuda_stream() : created(make_shared<bool>(false)) {
		CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
		created = true;
	}

	void cuda_stream::destroy() {
		if (created)
			CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
		created = false;
	}

	void cuda_stream::create() {
		if (!created)
			CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
		created = true;
	}

	void cuda_stream::sync() {
		CUDA_SAFE_CALL(cudaStreamSynchronize(_stream));
	}

	cuda_stream::cuda_stream(raw::cuda_types::cuda_stream &&rhs) noexcept
		: _stream(rhs._stream), created(rhs.created) {
		rhs._stream = nullptr;
		rhs.created = false;
	}

	cuda_stream &cuda_stream::operator=(raw::cuda_types::cuda_stream &&rhs) noexcept {
		destroy();
		_stream = rhs._stream;
		created = rhs.created;
		rhs._stream = nullptr;
		rhs.created = false;
		return *this;
	}

	cudaStream_t cuda_stream::stream() {
		return _stream;
	}

	cuda_stream::~cuda_stream() {
		if (created)
			cudaStreamDestroy(_stream);
	}
} // namespace raw::cuda_types