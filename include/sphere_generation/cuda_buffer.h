//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_BUFFER_H
#define SPACE_EXPLORER_CUDA_BUFFER_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace raw {
template<typename T>
class cuda_buffer {
private:
	T*	   ptr;
	size_t _size = 0;

public:
	explicit cuda_buffer(size_t size) : _size(size) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&ptr, _size));
	}
	cuda_buffer(cuda_buffer& rhs) {
		cuda_buffer(rhs._size);
		CUDA_SAFE_CALL(cudaMemcpy(ptr, rhs.ptr, _size));
	}
	cuda_buffer& operator=(const cuda_buffer& rhs) {
		CUDA_SAFE_CALL(cudaMemcpy(ptr, rhs.ptr, _size));
		return *this;
	}
	cuda_buffer(cuda_buffer&& rhs) noexcept : ptr(rhs.ptr), _size(rhs._size) {}
	cuda_buffer& operator=(cuda_buffer&& rhs) noexcept {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFree(ptr));
		}
		ptr	  = rhs.ptr;
		_size = rhs._size;
		return *this;
	}
	~cuda_buffer() {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFree(ptr));
		}
		ptr = nullptr;
	}
	T* get() const {
		return ptr;
	}
	void allocate(size_t size) {
		if (ptr) {
			CUDA_SAFE_CALL(cudaFree(ptr));
		}
		CUDA_SAFE_CALL(cudaMalloc((void**)&ptr, _size));
		_size = size;
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_CUDA_BUFFER_H
