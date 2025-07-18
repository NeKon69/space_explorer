//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
#define SPACE_EXPLORER_CUDA_FROM_GL_DATA_H

#include <cuda_gl_interop.h>
#include <helper_macros.h>

#include <format>
#include <source_location>

namespace raw {
class cuda_from_gl_data {
	// Meant to be used with ```new``` and deleted when cleanup starts

public:
	cudaGraphicsResource_t cuda_resource = nullptr;
	bool				   mapped		 = false;
	template<typename T>
	cuda_from_gl_data(T* data_out, size_t* amount_of_bytes, UI buffer_object) {
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsGLRegisterBuffer), &cuda_resource,
					   buffer_object, cudaGraphicsRegisterFlagsWriteDiscard);
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsMapResources), 1, &cuda_resource,
					   nullptr);
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsResourceGetMappedPointer),
					   (void**)&data_out, amount_of_bytes, cuda_resource);
	};
	void unmap();
	void map();

	~cuda_from_gl_data();
};
} // namespace raw
#endif // SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
