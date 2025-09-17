//
// Created by progamers on 8/5/25.
//

#pragma once
#include <cuda_gl_interop.h>

#include "common/fwd.h"
#include "device_types/cuda/error.h"
#include "device_types/cuda/resource_description.h"

namespace raw::device_types::cuda {
class surface {
private:
	cudaSurfaceObject_t surface_object = 0;
	bool				created		   = false;

public:
	template<typename T>
	explicit surface(resource_description<T> &description) {
		create(description);
	}
	template<typename T>

	void create(resource_description<T> &description) {
		if (!created) {
			CUDA_SAFE_CALL(cudaCreateSurfaceObject(surface_object, description));
		}
		created = true;
	}

	void destroy() {
		if (created) {
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(surface_object));
		}
		created = false;
	}

	~surface() {
		destroy();
	}

	cudaSurfaceObject_t &get() {
		return surface_object;
	}
};
} // namespace raw::device_types::cuda
