//
// Created by progamers on 9/6/25.
//

#ifndef SPACE_EXPLORER_CUDA_EXCEPTION_H
#define SPACE_EXPLORER_CUDA_EXCEPTION_H
#include <stdexcept>

namespace raw::device_types::cuda {
class cuda_exception : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};
} // namespace raw::device_types::cuda
#endif // SPACE_EXPLORER_CUDA_EXCEPTION_H
