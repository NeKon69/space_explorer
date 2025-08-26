//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_SURFACE_H
#define SPACE_EXPLORER_SURFACE_H
#include <cuda_gl_interop.h>

#include "common/fwd.h"
#include "cuda_types/error.h"
#include "cuda_types/resource_description.h"

namespace raw::cuda_types {
	template<typename T>
	class surface {
	private:
		cudaSurfaceObject_t surface_object = 0;
		bool created = false;

	public:
		explicit surface(raw::cuda_types::resource_description<T> &description) {
			create(description);
		}

		void create(raw::cuda_types::resource_description<T> &description) {
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
} // namespace raw::cuda_types
#endif // SPACE_EXPLORER_SURFACE_H