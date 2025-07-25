//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
#define SPACE_EXPLORER_CUDA_FROM_GL_DATA_H

#include <cuda_gl_interop.h>
#include <helper_macros.h>

#include <format>
#include <source_location>

#include "custom_deleters.h"

namespace raw {
template<typename T>
class cuda_from_gl_data {
	// Meant to be used with ```new``` (or shared-ptr) and deleted when cleanup starts

public:
	cudaGraphicsResource_t cuda_resource = nullptr;
	T*					   data			 = nullptr;
	bool				   mapped		 = false;
	cuda_from_gl_data()					 = default;
	cuda_from_gl_data(size_t* amount_of_bytes, UI buffer_object) {
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_resource, buffer_object,
													cudaGraphicsRegisterFlagsWriteDiscard));
		CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_resource, nullptr));
		CUDA_SAFE_CALL(
			cudaGraphicsResourceGetMappedPointer((void**)&data, amount_of_bytes, cuda_resource));
		mapped = true;
	};
	cuda_from_gl_data& operator=(cuda_from_gl_data&& rhs) noexcept {
		mapped			  = rhs.mapped;
		rhs.mapped		  = false;
		cuda_resource	  = rhs.cuda_resource;
		data			  = rhs.data;
		rhs.cuda_resource = nullptr;
		rhs.data		  = nullptr;
		return *this;
	}
	[[nodiscard]] T* get_data() const {
		return data;
	}
	void unmap() {
		if (mapped)
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_resource, nullptr));
		mapped = false;
	}
	void map() {
		if (!mapped)
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_resource, nullptr));
		mapped = true;
	}

	~cuda_from_gl_data() {
		unmap();
		if (cuda_resource)
			cudaGraphicsUnregisterResource(cuda_resource);
	}
};
} // namespace raw
#endif // SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
