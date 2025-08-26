//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_ERROR_H
#define SPACE_EXPLORER_ERROR_H
#include <format>
#include <source_location>
#include <stdexcept>

#define CUDA_SAFE_CALL(call)                                                                            \
	do {                                                                                                \
		cudaError_t error = call;                                                                       \
		if (error != cudaSuccess) {                                                                     \
			const char* msg		 = cudaGetErrorName(error);                                             \
			const char* msg_name = cudaGetErrorString(error);                                           \
			throw std::runtime_error(std::format(                                                       \
				"[Error] Function {} failed with error: {} and description: {} in file: {} on line {}", \
				#call, msg, msg_name, std::source_location::current().file_name(),                      \
				std::source_location::current().line()));                                               \
		}                                                                                               \
	} while (0)

#endif // SPACE_EXPLORER_ERROR_H