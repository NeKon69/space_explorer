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
template<typename T>
class cuda_buffer {
private:
	raw::shared_ptr<cuda_stream> data_stream;
	T*							 ptr;
	size_t						 _size = 0;

public:
	explicit cuda_buffer(size_t size) : _size(size), data_stream(raw::make_shared<cuda_stream>()) {
		CUDA_SAFE_CALL(cudaMallocAsync((void**)&ptr, _size, data_stream->stream()));
	}
	cuda_buffer(cuda_buffer& rhs) : data_stream(rhs.data_stream) {
		cuda_buffer(rhs._size);
		CUDA_SAFE_CALL(cudaMemcpyAsync(ptr, rhs.ptr, _size, data_stream->stream()));
	}
	cuda_buffer& operator=(const cuda_buffer& rhs) {
		if (this == rhs) {
			return *this;
		}
		CUDA_SAFE_CALL(cudaMemcpyAsync(ptr, rhs.ptr, rhs._size, data_stream->stream()));
		_size		= rhs._size;
		data_stream = rhs.data_stream;
		return *this;
	}
	cuda_buffer(cuda_buffer&& rhs) noexcept
		: ptr(rhs.ptr), _size(rhs._size), data_stream(make_shared<cuda_stream>()) {}
	cuda_buffer& operator=(cuda_buffer&& rhs) noexcept {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		ptr	  = rhs.ptr;
		_size = rhs._size;
		return *this;
	}
	~cuda_buffer() {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		ptr = nullptr;
	}
	T* get() const {
		cudaStreamSynchronize(data_stream->stream());
		return ptr;
	}
	void allocate(size_t size) {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
		}
		CUDA_SAFE_CALL(cudaMallocAsync((void**)&ptr, _size, data_stream->stream()));
		_size = size;
	}
	void set_data(void* data, size_t size) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(ptr, data, size, data_stream->stream()));
	}
	void free() {
		CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream));
		ptr = nullptr;
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_CUDA_BUFFER_H
