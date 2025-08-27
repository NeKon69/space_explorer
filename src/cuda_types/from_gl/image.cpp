//
// Created by progamers on 8/5/25.
//
#include "cuda_types/from_gl/image.h"

namespace raw::cuda_types::from_gl {
image::image(raw::UI texture_id)
	: raw::cuda_types::resource(cudaGraphicsGLRegisterImage, texture_id, GL_TEXTURE_2D,
	                            cudaGraphicsRegisterFlagsSurfaceLoadStore) {
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, get_resource(), 0, 0));
}

void image::set_data(raw::UI texture_id) {
	create(cudaGraphicsGLRegisterImage, texture_id, GL_TEXTURE_2D,
	       cudaGraphicsRegisterFlagsSurfaceLoadStore);
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, get_resource(), 0, 0));
}

cudaArray_t image::get() {
	map();
	return array;
}
} // namespace raw::cuda_types::from_gl
