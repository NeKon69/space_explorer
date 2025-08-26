//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_BUFFER_H
#define SPACE_EXPLORER_CUDA_BUFFER_H

#include <cuda_runtime.h>

#include "common/fwd.h"
#include "cuda_types/error.h"
#include "cuda_types/fwd.h"
#include "cuda_types/stream.h"

namespace raw::cuda_types {
// This motherfucker right here, yes, this one, he is the fucking ugliest part of my code, it
// sucks, it's ugly, and also... IDK
// I need to think of some better design for that shit
template<typename T>
class cuda_buffer {
private:
	// Boom! Genius use of streams
	size_t						 _size = 0;
	raw::shared_ptr<cuda_stream> data_stream;
	T							*ptr		   = nullptr;
	bool						 manual_stream = false;

	void _memcpy(T *dst, T *src, size_t size, cudaMemcpyKind kind) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, size, kind, data_stream->stream()));
	}

public:
	cuda_buffer() = default;

	static cuda_buffer create(const size_t size) {
		// so there'll be only one stream for all buffers. nice!
		static raw::shared_ptr<cuda_stream> _stream = raw::make_shared<cuda_stream>();
		return cuda_buffer<T>(size, _stream);
	}

	explicit cuda_buffer(const size_t size)
		: _size(size), data_stream(raw::make_shared<cuda_stream>()) {
		CUDA_SAFE_CALL(cudaMallocAsync((void **)&ptr, _size, data_stream->stream()));
	}

	cuda_buffer(const size_t size, raw::shared_ptr<cuda_stream> stream, const bool manual = false)
		: _size(size), data_stream(std::move(stream)) {
		CUDA_SAFE_CALL(cudaMallocAsync((void **)&ptr, _size, data_stream->stream()));
		manual_stream = manual;
	}

	cuda_buffer(const cuda_buffer &rhs) : data_stream(rhs.data_stream) {
		cuda_buffer(rhs._size);
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDeviceToDevice);
	}

	cuda_buffer &operator=(const cuda_buffer &rhs) {
		if (this == rhs) {
			return *this;
		}
		_memcpy(ptr, rhs.ptr, rhs._size, cudaMemcpyDeviceToDevice);
		_size		= rhs._size;
		data_stream = rhs.data_stream;
		return *this;
	}

	cuda_buffer(cuda_buffer &&rhs) noexcept
		: ptr(rhs.ptr), _size(rhs._size), data_stream(std::move(rhs.data_stream)) {}

	cuda_buffer &operator=(cuda_buffer &&rhs) noexcept {
		free();
		ptr				= rhs.ptr;
		_size			= rhs._size;
		data_stream		= std::move(rhs.data_stream);
		rhs.ptr			= nullptr;
		rhs._size		= 0;
		rhs.data_stream = nullptr;
		return *this;
	}

	~cuda_buffer() {
		free();
		ptr			= nullptr;
		data_stream = nullptr;
	}

	T *get() const {
		return ptr;
	}

	void allocate(size_t size) {
		free();
		CUDA_SAFE_CALL(cudaMallocAsync(&ptr, size, data_stream->stream()));
		_size = size;
	}

	void set_data(T *data, size_t size, cudaMemcpyKind kind = cudaMemcpyDefault,
				  cudaMemcpyOrder order = cudaMemcpyOrder::cudaMemcpy2to1) {
		using enum cudaMemcpyOrder;
		if (size > _size) {
			std::cout
				<< "[Warning] Requested amount of bytes to copy was more when currently allocated\n";
			size = _size;
		}
		if (order == cudaMemcpy2to1) {
			_memcpy(ptr, data, size, kind);
		} else {
			_memcpy(data, ptr, size, kind);
		}
	}

	void free() {
		if (_size != 0) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
			_size = 0;
		}
	}

	explicit operator bool() const {
		return ptr != nullptr;
	}

	void zero_data(size_t amount) {
		cudaMemsetAsync(ptr, 0, amount, data_stream->stream());
	}
};
} // namespace raw::cuda_types

#endif // SPACE_EXPLORER_CUDA_BUFFER_H