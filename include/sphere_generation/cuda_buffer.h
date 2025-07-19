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
		cudaError_t result = cudaMalloc((void**)&ptr, _size);
		if (result != cudaSuccess) {
			const char* msg = cudaGetErrorName(result);
			throw std::runtime_error(
				std::format("[Error] Function {} failed with error: {} in file: {} on line {}",
							"cudaMalloc", msg, std::source_location::current().file_name(),
							std::source_location::current().line()));
		}
	}
	cuda_buffer(cuda_buffer& rhs) {
		cuda_buffer(rhs._size);
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaMemcpy), ptr, rhs.ptr, _size);
	}
	cuda_buffer& operator=(const cuda_buffer& rhs) {
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaMemcpy), ptr, rhs.ptr, _size);
		return *this;
	}
	cuda_buffer(cuda_buffer&& rhs) noexcept : ptr(rhs.ptr), _size(rhs._size) {}
	cuda_buffer& operator=(cuda_buffer&& rhs) {
		if (ptr)
			CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaFree), ptr);
		ptr	  = rhs.ptr;
		_size = rhs._size;
		return *this;
	}
	~cuda_buffer() {
		if (ptr)
			CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaFree), ptr);
		ptr = nullptr;
	}
	T* get() const {
		return ptr;
	}
	void allocate(size_t size) {
		if (ptr)
			CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaFree), ptr);
        cudaError_t result = cudaMalloc((void**)&ptr, size);
        if (result != cudaSuccess) {
            const char* msg = cudaGetErrorName(result);
            throw std::runtime_error(
                    std::format("[Error] Function {} failed with error: {} in file: {} on line {}",
                                "cudaMalloc", msg, std::source_location::current().file_name(),
                                std::source_location::current().line()));
        }
		_size = size;
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_CUDA_BUFFER_H
