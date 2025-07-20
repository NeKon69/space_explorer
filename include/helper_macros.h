//
// Created by progamers on 6/27/25.
//

#ifndef SPACE_EXPLORER_HELPER_MACROS_H
#define SPACE_EXPLORER_HELPER_MACROS_H
#include <cuda_runtime.h>

#include <chrono>
#include <exception>
#include <format>
#include <iostream>
#include <source_location>
#include <vector>

namespace raw {
#define PASSIVE_VALUE static constexpr auto
using std_clock = std::chrono::high_resolution_clock;
using UI		= unsigned int;
template<typename T>
using vec = std::vector<T>;
#define CUDA_SAFE_CALL(call)                                                                    \
	do {                                                                                        \
		cudaError_t error = call;                                                               \
		if (error != cudaSuccess) {                                                             \
			const char* msg = cudaGetErrorName(error);                                          \
			throw std::runtime_error(                                                           \
				std::format("[Error] Function {} failed with error: {} in file: {} on line {}", \
							#call, msg, std::source_location::current().file_name(),            \
							std::source_location::current().line()));                           \
		}                                                                                       \
	} while (0)

} // namespace raw

#endif // SPACE_EXPLORER_HELPER_MACROS_H
