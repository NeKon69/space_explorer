//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_BUFFER_H
#define SPACE_EXPLORER_CUDA_BUFFER_H

#include <cuda_runtime.h>

namespace raw {
template<typename T>
class cuda_buffer {
private:
	T*	   ptr	 = nullptr;
	size_t _size = 0;

public:
	explicit cuda_buffer(size_t size) : _size(size) {
		cudaMalloc(&ptr, size);
	}
	cuda_buffer(cuda_buffer& rhs) {
		cuda_buffer(rhs._size);
		cudaMemcpy(ptr, rhs.ptr, _size);
	}
	cuda_buffer& operator=(const cuda_buffer& rhs) {
		cudaMemcpy(ptr, rhs.ptr, _size);
		return *this;
	}
	cuda_buffer(cuda_buffer&& rhs) noexcept : ptr(rhs.ptr), _size(rhs._size) {}
	cuda_buffer& operator=(cuda_buffer&& rhs) {
		if (ptr)
			cudaFree(ptr);
		ptr	  = rhs.ptr;
		_size = rhs._size;
		return *this;
	}
	~cuda_buffer() {
		if (ptr)
			cudaFree(ptr);
		ptr = nullptr;
	}
	T* get() const {
		return ptr;
	}
	void allocate(size_t size) {
		if (ptr)
			cudaFree(ptr);
		cudaMalloc(&ptr, size);
		_size = size;
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_CUDA_BUFFER_H
