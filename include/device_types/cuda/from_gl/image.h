//
// Created by progamers on 8/5/25.
//

#pragma once
#include "device_types/cuda/resource.h"

namespace raw::device_types::cuda::from_gl {
class image : public resource {
private:
	cudaArray_t array = nullptr;

public:
	using resource::resource;

	// If used default constructor and didn't set the data manually you are screwed (let it be UB)
	image() = default;

	explicit image(uint32_t texture_id);

	void set_data(uint32_t texture_id);

	cudaArray_t& get();
};
} // namespace raw::cuda::from_gl

