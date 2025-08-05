//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_IMAGE_H
#define SPACE_EXPLORER_IMAGE_H
#include <cuda_gl_interop.h>

#include "cuda_types/resource.h"
namespace raw::cuda::gl {
class image : raw::cuda::resource {
private:
	cudaArray_t array;

public:
	image(raw::UI vbo);
	cudaArray_t get();
};
} // namespace raw::cuda::gl

#endif // SPACE_EXPLORER_IMAGE_H
