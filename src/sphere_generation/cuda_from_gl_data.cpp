//
// Created by progamers on 7/18/25.
//
#include "sphere_generation/cuda_from_gl_data.h"
namespace raw {
void cuda_from_gl_data::unmap() {
	CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsUnmapResources), 1, &cuda_resource, nullptr);
}
void cuda_from_gl_data::map() {
	CUDA_SAFE_CALL(GET_FUNC_AND_FUNC_NAME(cudaGraphicsMapResources), 1, &cuda_resource, nullptr);
}
cuda_from_gl_data::~cuda_from_gl_data() {
	cudaGraphicsUnregisterResource(cuda_resource);
}
} // namespace raw