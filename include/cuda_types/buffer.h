//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_BUFFER_H
#define SPACE_EXPLORER_CUDA_BUFFER_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_macros.h"
#include "stream.h"
namespace raw {
enum class cudaMemcpyOrder { cudaMemcpy1to2, cudaMemcpy2to1 };
template<typename T>
class cuda_buffer {
private:
	// Boom! Genius use of streams
	size_t						 _size = 0;
	raw::shared_ptr<cuda_stream> data_stream;
	T*							 ptr		   = nullptr;
	bool						 manual_stream = false;

	void _memcpy(T* dst, T* src, size_t size, cudaMemcpyKind kind) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, size, kind, data_stream->stream()));
	}

public:
	cuda_buffer() = default;
	static cuda_buffer create(size_t size) {
		// so there'll be only one stream for all buffers. nice!
		static raw::shared_ptr<cuda_stream> _stream = raw::make_shared<cuda_stream>();
		return cuda_buffer<T>(size, _stream);
	}
	explicit cuda_buffer(size_t size) : _size(size), data_stream(raw::make_shared<cuda_stream>()) {
		CUDA_SAFE_CALL(cudaMallocAsync((void**)&ptr, _size, data_stream->stream()));
	}
	cuda_buffer(size_t size, raw::shared_ptr<cuda_stream> stream, bool manual = false)
		: _size(size), data_stream(std::move(stream)) {
		CUDA_SAFE_CALL(cudaMallocAsync((void**)&ptr, _size, data_stream->stream()));
		manual_stream = manual;
	}
	cuda_buffer(const cuda_buffer& rhs) : data_stream(rhs.data_stream) {
		cuda_buffer(rhs._size);
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDeviceToDevice);
	}
	cuda_buffer& operator=(const cuda_buffer& rhs) {
		if (this == rhs) {
			return *this;
		}
		_memcpy(ptr, rhs.ptr, rhs._size, cudaMemcpyDeviceToDevice);
		_size		= rhs._size;
		data_stream = rhs.data_stream;
		return *this;
	}
	cuda_buffer(cuda_buffer&& rhs) noexcept
		: ptr(rhs.ptr), _size(rhs._size), func data_stream(std::move(rhs.data_stream)) {}
	cuda_buffer& operator=(cuda_buffer&& rhs) noexcept {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		ptr				= rhs.ptr;
		_size			= rhs._size;
		data_stream		= std::move(rhs.data_stream);
		rhs.ptr			= nullptr;
		rhs._size		= 0;
		rhs.data_stream = nullptr;
		return *this;
	}
	~cuda_buffer() {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		ptr			= nullptr;
		_size		= 0;
		data_stream = nullptr;
	}
	T* get() const {
		if (!manual_stream)
			data_stream->sync();
		return ptr;
	}
	void allocate(size_t size) {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		CUDA_SAFE_CALL(cudaMallocAsync(&ptr, size, data_stream->stream()));
		_size = size;
	}
	void set_data(T* data, size_t size, cudaMemcpyKind kind = cudaMemcpyDefault,
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
		CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		ptr	  = nullptr;
		_size = 0;
	}
	explicit operator bool() const {
		return ptr != nullptr;
	}
	void zero_data(size_t amount) {
		cudaMemsetAsync(ptr, 0, amount, data_stream->stream());
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_CUDA_BUFFER_H
