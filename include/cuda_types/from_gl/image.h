//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_IMAGE_H
#define SPACE_EXPLORER_IMAGE_H
#include <cuda_gl_interop.h>

#include "../error.h"
#include "cuda_types/fwd.h"
#include "cuda_types/resource.h"

namespace raw::cuda_types::from_gl {
	class image : raw::cuda_types::resource {
	private:
		cudaArray_t array;

	public:
		using raw::cuda_types::resource::resource;

		// If used default constructor and didn't set the data manually you are screwed (let it be UB)
		image() = default;

		image(raw::UI texture_id);

		void set_data(raw::UI texture_id);

		cudaArray_t get();
	};
} // namespace raw::cuda_types::from_gl

#endif // SPACE_EXPLORER_IMAGE_H