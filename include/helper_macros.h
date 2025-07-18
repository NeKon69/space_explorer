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
#define GET_FUNC_AND_FUNC_NAME(x) x, #x
template<typename Func, typename... Ts>
void CUDA_SAFE_CALL(Func&& func, std::string func_name, Ts&&... ts) {
	cudaError_t result = std::forward<Func>(func)(std::forward<Ts>(ts)...);
	if (result != cudaSuccess) {
		const char* msg = cudaGetErrorName(result);
		throw std::runtime_error(std::format(
			"{} failed with error: {} in file: {} on line {}", func_name, msg,
			std::source_location::current().file_name(), std::source_location::current().line()));
	}
}

} // namespace raw

#endif // SPACE_EXPLORER_HELPER_MACROS_H
