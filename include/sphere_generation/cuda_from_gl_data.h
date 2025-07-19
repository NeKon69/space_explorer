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
    template<typename T>
class cuda_from_gl_data {
	// Meant to be used with ```new``` (or shared-ptr) and deleted when cleanup starts

public:
	cudaGraphicsResource_t cuda_resource = nullptr;
	T*				   data;
	bool				   mapped		 = false;
	cuda_from_gl_data()					 = default;
	cuda_from_gl_data(size_t* amount_of_bytes, UI buffer_object) {
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsGLRegisterBuffer), &cuda_resource,
					   buffer_object, cudaGraphicsRegisterFlagsWriteDiscard);
		CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsMapResources), 1, &cuda_resource,
					   nullptr);
	    cudaGraphicsResourceGetMappedPointer( (void**)&data,
					   amount_of_bytes, cuda_resource);
		if (!data)
			throw std::runtime_error(std::format(
				"[Error] Could not get pointer to opengl data in file \"{}\" in function \"{}\" on line {}",
				__FILE_NAME__, __FUNCTION__, __LINE__ - 5));
		mapped = true;
	};
	[[nodiscard]] T* get_data() const {
		return data;
	}
	void unmap() {
        if(mapped)
            CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsUnmapResources), 1, &cuda_resource, nullptr);
        mapped = false;
    }
	void map() {
        if(!mapped)
            CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsMapResources), 1, &cuda_resource, nullptr);
        mapped = true;
    }

    ~cuda_from_gl_data() {
        if(mapped) unmap();
        cudaGraphicsUnregisterResource(cuda_resource);
    }
};
} // namespace raw
#endif // SPACE_EXPLORER_CUDA_FROM_GL_DATA_H
