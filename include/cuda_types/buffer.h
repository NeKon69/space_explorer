//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_BUFFER_H
#define SPACE_EXPLORER_CUDA_BUFFER_H

#include <cuda/std/__ranges/data.h>
#include <cuda_runtime.h>

#include "common/fwd.h"
#include "cuda_types/error.h"
#include "cuda_types/fwd.h"
#include "cuda_types/stream.h"

namespace raw::cuda_types {
// This motherfucker right here, yes, this one, he is the fucking ugliest part of my code, it
// sucks, it's ugly, and also... IDK
// I need to think of some better design for that shit
template<typename T, side Side>
class cuda_buffer {
private:
	// Boom! Genius use of streams
	size_t						 _size = 0;
	std::shared_ptr<cuda_stream> data_stream;
	T							*ptr		   = nullptr;
	bool						 manual_stream = false;

	void _memcpy(T *dst, T *src, size_t size, cudaMemcpyKind kind) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, size, kind, data_stream->stream()));
	}
	void alloc()
		requires(Side == side::device)
	{
		CUDA_SAFE_CALL(cudaMallocAsync(&ptr, _size, data_stream->stream()));
	}
	void alloc()
		requires(Side == side::host)
	{
		CUDA_SAFE_CALL(cudaMallocHost(&ptr, _size));
	}

public:
	cuda_buffer() = default;

	__device__ __host__ T &operator*()
		requires(Side == side::host)
	{
		return *ptr;
	}

	static cuda_buffer create(const size_t size) {
		// so there'll be only one stream for all buffers. nice!
		static std::shared_ptr<cuda_stream> _stream = std::make_shared<cuda_stream>();
		return cuda_buffer<T>(size, _stream);
	}

	explicit cuda_buffer(const size_t size) : _size(size) {
		if constexpr (Side == side::device) {
			data_stream = std::make_shared<cuda_stream>();
		}
		alloc();
	}

	cuda_buffer(const size_t size, std::shared_ptr<cuda_stream> stream, const bool manual = false)
		: _size(size), data_stream(std::move(stream)) {
		alloc();
		manual_stream = manual;
	}

	cuda_buffer(const cuda_buffer &rhs) : _size(rhs._size), data_stream(rhs.data_stream) {
		alloc();
		if constexpr (Side == side::device) {
			_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDeviceToDevice);
		} else {
			_memcpy(ptr, rhs.ptr, _size, cudaMemcpyHostToHost);
		}
	}

	cuda_buffer &operator=(const cuda_buffer &rhs) {
		if (this == &rhs) {
			return *this;
		}
		cuda_buffer temp(rhs);
		std::swap(ptr, temp.ptr);
		std::swap(_size, temp._size);
		std::swap(data_stream, temp.data_stream);
		return *this;
	}

	cuda_buffer(cuda_buffer &&rhs) noexcept
		: ptr(rhs.ptr), _size(rhs._size), data_stream(std::move(rhs.data_stream)) {
		rhs.ptr	  = nullptr;
		rhs._size = 0;
	}

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
		_size = size;
		alloc();
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
			if constexpr (Side == side::device) {
				CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
			} else {
				CUDA_SAFE_CALL(cudaFreeHost(ptr));
			}
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