//
// Created by progamers on 8/5/25.
//
#include "cuda_types/from_gl/image.h"
namespace raw::cuda::gl {
image::image(raw::UI vbo)
	: raw::cuda::resource(cudaGraphicsGLRegisterImage, vbo, GL_TEXTURE_2D,
						  cudaGraphicsRegisterFlagsSurfaceLoadStore) {
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, get_resource(), 0, 0));
}
cudaArray_t image::get() {
	map();
	return array;
}
} // namespace raw::cuda::gl