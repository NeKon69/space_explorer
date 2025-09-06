//
// Created by progamers on 9/6/25.
//

#ifndef SPACE_EXPLORER_CUDA_EXCEPTION_H
#define SPACE_EXPLORER_CUDA_EXCEPTION_H
#include <stdexcept>

#include "cuda_types/fwd.h"
namespace raw::cuda_types {
class cuda_exception : public std::exception{
private:
	std::string message;

public:
	explicit cuda_exception(std::string message) : message(std::move(message)) {}
	[[nodiscard]] const char* what() const noexcept override {
		return message.c_str();
	}
};
} // namespace raw::cuda_types
#endif // SPACE_EXPLORER_CUDA_EXCEPTION_H
