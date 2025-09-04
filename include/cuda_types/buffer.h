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
	size_t						 _size = 0;
	std::shared_ptr<cuda_stream> data_stream;
	T							*ptr = nullptr;

	void _memcpy(T *dst, const T *src, size_t size, cudaMemcpyKind kind) {
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
	cuda_buffer() : data_stream(std::make_shared<cuda_stream>()) {}

	__host__ T &operator*()
		requires(Side == side::host)
	{
		return *ptr;
	}
	const __host__ T &operator*() const
		requires(Side == side::device)
	{
		return *ptr;
	}

	/** @deprecated This lost its purpose with introduction of a stream to each part of project (with gpgpu calculations)
	 */
	static cuda_buffer create(const size_t size) {
		static std::shared_ptr<cuda_stream> _stream = std::make_shared<cuda_stream>();
		return cuda_buffer<T>(size, _stream);
	}

	explicit cuda_buffer(const size_t size) : _size(size) {
		data_stream = std::make_shared<cuda_stream>();
		alloc();
	}

	cuda_buffer(const size_t size, std::shared_ptr<cuda_stream> stream)
		: _size(size), data_stream(std::move(stream)) {
		alloc();
	}

	template<side Side_>
	explicit cuda_buffer(const cuda_buffer<T, Side_> &rhs)
		: _size(rhs._size), data_stream(rhs.data_stream) {
		alloc();
		// Only works when the address is unified, and since this is exactly what we do here, it's
		// fine
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDefault);
	}

	cuda_buffer &operator=(const cuda_buffer &rhs) {
		if (this == &rhs) {
			return *this;
		}
		data_stream = rhs.data_stream;
		_size		= rhs._size;

		alloc();
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDefault);

		return *this;
	}

	cuda_buffer(cuda_buffer &&rhs) noexcept
		: _size(rhs._size), data_stream(std::move(rhs.data_stream)), ptr(rhs.ptr) {
		rhs.ptr	  = nullptr;
		rhs._size = 0;
	}

	cuda_buffer &operator=(cuda_buffer &&rhs) noexcept {
		free();
		data_stream = std::move(rhs.data_stream);

		ptr		  = rhs.ptr;
		rhs.ptr	  = nullptr;
		_size	  = rhs._size;
		rhs._size = 0;

		return *this;
	}

	~cuda_buffer() {
		free();
		ptr			= nullptr;
		data_stream = nullptr;
		_size		= 0;
	}

	T *get() const {
		return ptr;
	}

	void allocate(size_t size) {
		free();
		_size = size;
		alloc();
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

	void memset(void* _ptr, size_t size, cudaMemcpyKind kind) {
		cudaMemcpyAsync(ptr, _ptr, size, kind, data_stream->stream());
	}

	explicit operator bool() const {
		return ptr != nullptr;
	}

	void zero_data(size_t amount) const {
		cudaMemsetAsync(ptr, 0, amount, data_stream->stream());
	}
};
} // namespace raw::cuda_types

#endif // SPACE_EXPLORER_CUDA_BUFFER_H