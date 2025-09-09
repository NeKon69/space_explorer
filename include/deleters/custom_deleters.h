//
// Created by progamers on 7/23/25.
//

#pragma once
#include <cuda_runtime.h>

#include "common/fwd.h"

namespace raw::deleters {
struct gl_array {
	explicit gl_array(UI const* data);
};
struct gl_buffer {
	explicit gl_buffer(UI const* data);
};
struct gl_texture {
	explicit gl_texture(UI const* data);
};
struct cuda_gl_data {
	explicit cuda_gl_data(cudaGraphicsResource_t* data) {
		if (data) {
			cudaGraphicsUnregisterResource(*data);
		}

		delete data;
	}
};
} // namespace raw::deleters

