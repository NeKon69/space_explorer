//
// Created by progamers on 7/18/25.
//

#pragma once

#include <cuda_runtime.h>

#include "common/fwd.h"
#include "device_types/cuda/error.h"
#include "device_types/cuda/fwd.h"
#include "device_types/cuda/stream.h"

namespace raw::device_types::cuda {
// This motherfucker right here, yes, this one, he is the fucking ugliest part of my code, it
// sucks, it's ugly, and also... IDK
// I need to think of some better design for that shit
template<typename T, side Side>
class buffer {
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
	buffer(std::shared_ptr<cuda_stream> stream) : data_stream(stream) {}

	__host__ T &operator*()
		requires(Side == side::host)
	{
		return *ptr;
	}

	/** @deprecated This lost its purpose with introduction of a stream to each part of project
	 * (with gpgpu calculations)
	 */
	static buffer create(const size_t size) {
		static std::shared_ptr<cuda_stream> _stream = std::make_shared<cuda_stream>();
		return buffer<T>(size, _stream);
	}

	explicit buffer(const size_t size) : _size(size) {
		data_stream = std::make_shared<cuda_stream>();
		alloc();
	}

	buffer(const size_t size, std::shared_ptr<cuda_stream> stream)
		: _size(size), data_stream(std::move(stream)) {
		alloc();
	}

	template<side Side_>
	explicit buffer(const buffer<T, Side_> &rhs) : _size(rhs._size), data_stream(rhs.data_stream) {
		alloc();
		// Only works when the address is unified, and since this is exactly what we do here, it's
		// fine
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDefault);
	}

	buffer &operator=(const buffer &rhs) {
		if (this == &rhs) {
			return *this;
		}
		data_stream = rhs.data_stream;
		_size		= rhs._size;

		alloc();
		_memcpy(ptr, rhs.ptr, _size, cudaMemcpyDefault);

		return *this;
	}

	buffer(buffer &&rhs) noexcept
		: _size(rhs._size), data_stream(std::move(rhs.data_stream)), ptr(rhs.ptr) {
		rhs.ptr	  = nullptr;
		rhs._size = 0;
	}

	buffer &operator=(buffer &&rhs) noexcept {
		free();
		data_stream = std::move(rhs.data_stream);

		ptr		  = rhs.ptr;
		rhs.ptr	  = nullptr;
		_size	  = rhs._size;
		rhs._size = 0;

		return *this;
	}

	~buffer() {
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
			try {
				if constexpr (Side == side::device) {
					CUDA_SAFE_CALL(cudaFreeAsync(ptr, data_stream->stream()));
				} else {
					CUDA_SAFE_CALL(cudaFreeHost(ptr));
				}
			} catch (const cuda_exception &e) {
				std::cerr << std::format("[CRITICAL] Error Occured In Free Function. \n{}",
										 e.what())
						  << std::endl;
			}
			_size = 0;
		}
	}

	void memset(const void *_ptr, size_t size, cudaMemcpyKind kind) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(ptr, _ptr, size, kind, data_stream->stream()));
	}

	explicit operator bool() const {
		return ptr != nullptr;
	}

	void zero_data(size_t amount) const {
		CUDA_SAFE_CALL(cudaMemsetAsync(ptr, 0, amount, data_stream->stream()));
	}
	void set_stream(std::shared_ptr<cuda_stream> stream) {
		data_stream = std::move(stream);
	}

	void memcpy(const void *_ptr, size_t size, size_t offset, cudaMemcpyKind kind) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(ptr + offset, _ptr, size, kind, data_stream->stream()));
	}
	[[nodiscard]] size_t get_size() const {
		return _size;
	}
};
} // namespace raw::device_types::cuda

