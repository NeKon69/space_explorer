//
// Created by progamers on 8/5/25.
//

#pragma once
#include <cuda_runtime.h>

#include <cstring>

#include "device_types/cuda/fwd.h"

namespace raw::device_types::cuda {
namespace resource_types {
struct array {
	cudaResourceType res_type = cudaResourceTypeArray;
};
} // namespace resource_types

template<typename T>
class resource_description {
private:
	cudaResourceDesc description;

public:
	resource_description() {
		std::memset(&description, 0, sizeof(description));
		description.resType = T::res_type;
	}
	std::enable_if_t<std::is_same_v<T, resource_types::array>, void> set_array(cudaArray_t& array) {
		description.res.array.array = array;
	}
};
} // namespace raw::device_types::cuda
