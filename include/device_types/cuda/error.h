//
// Created by progamers on 8/26/25.
//

#pragma once
#include <format>
#include <source_location>
#include <stdexcept>

#include "device_types/cuda/exception.h"
#define CUDA_SAFE_CALL(call)                                                                            \
	do {                                                                                                \
		cudaError_t error = call;                                                                       \
		if (error != cudaSuccess) {                                                                     \
			const char* msg		 = cudaGetErrorName(error);                                             \
			const char* msg_name = cudaGetErrorString(error);                                           \
			throw raw::device_types::cuda::cuda_exception(std::format(                                                     \
				"[Error] Function {} failed with error: {} and description: {} in file: {} on line {}", \
				#call, msg, msg_name, std::source_location::current().file_name(),                      \
				std::source_location::current().line()));                                               \
		}                                                                                               \
	} while (0)

