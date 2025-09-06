//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
#define SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
#include <cuda_gl_interop.h>

#include "common/fwd.h"
#include "cuda_types/error.h"
#include "cuda_types/from_gl/fwd.h"
#include "cuda_types/fwd.h"
#include "cuda_types/resource.h"

namespace raw::cuda_types::from_gl {
template<typename T>
class buffer : public resource {
	// Meant to be used with ```new``` (or shared-ptr) and deleted when cleanup starts

private:
	T*	   data	 = nullptr;
	size_t bytes = 0;

public:
	using resource::resource;
	buffer() = default;

	buffer(size_t* amount_of_bytes, UI buffer_object, std::shared_ptr<cuda_stream> stream)
		: resource(cudaGraphicsGLRegisterBuffer, stream, buffer_object,
				   cudaGraphicsRegisterFlagsWriteDiscard) {
		map();
		CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&data, &bytes, get_resource()));
		if (amount_of_bytes)
			*amount_of_bytes = bytes;
		unmap();
	}
	buffer(const buffer&)			 = delete;
	buffer& operator=(const buffer&) = delete;
	buffer(buffer&& rhs) noexcept : resource(std::move(rhs)), data(rhs.data), bytes(rhs.bytes) {
		rhs.data  = nullptr;
		rhs.bytes = 0;
	}
	buffer& operator=(buffer&& rhs) noexcept {
		if (this == &rhs) {
			return *this;
		}
		resource::operator=(std::move(rhs));
		data  = rhs.data;
		bytes = rhs.bytes;

		rhs.data  = nullptr;
		rhs.bytes = 0;

		return *this;
	}
	[[nodiscard]] T* get_data() const {
		return data;
	}

	~buffer() override = default;
};
} // namespace raw::cuda_types::from_gl
#endif // SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
