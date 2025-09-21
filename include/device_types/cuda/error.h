//
// Created by progamers on 8/26/25.
//

#pragma once
#include <format>
#include <source_location>
#include <stdexcept>

#include "device_types/cuda/exception.h"
#define __CHECK_CUDA_ERROR(error, call, line)                                                       \
	if ((error) != cudaSuccess) {                                                                   \
		const char* msg		 = cudaGetErrorName(error);                                             \
		const char* msg_name = cudaGetErrorString(error);                                           \
		throw raw::device_types::cuda::cuda_exception(std::format(                                  \
			"[Error] Function {} failed with error: {} and description: {} in file: {} on line {}", \
			#call, msg, msg_name, std::source_location::current().file_name(), line));              \
	}

#define CUDA_SAFE_CALL(call)                                                    \
	do {                                                                        \
		cudaError_t error = call;                                               \
		__CHECK_CUDA_ERROR(error, call, std::source_location::current().line()) \
	} while (0)

#define CHECK_CUDA_ERROR()                                                            \
	do {                                                                              \
		cudaError_t error = cudaGetLastError();                                       \
		__CHECK_CUDA_ERROR(error, "UNKNOWN (check line filename/line for more info)", \
						   std::source_location::current().line() - 1)                \
	} while (0)

#define CUDA_FAIL_ON_ERROR(error)                                                     \
	do {                                                                              \
		__CHECK_CUDA_ERROR(error, "UNKNOWN (check line filename/line for more info)", \
						   std::source_location::current().line())                    \
	} while (0)