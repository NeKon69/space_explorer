//
// Created by progamers on 9/6/25.
//

#pragma once
#include <CL/cl.h>

#include <stdexcept>
#include <variant>

namespace raw::device_types {
enum class backend { CUDA, OPENCL };

template<typename T, backend B>
struct backend_traits;

template<typename T>
struct backend_traits<T, backend::CUDA> {
	using native_handle_type = T;
};

template<typename T>
struct backend_traits<T, backend::OPENCL> {
	using native_handle_type = cl_mem;
};

template<typename T, backend B>
using native_handle_t = backend_traits<T, B>::native_handle_type;

// Template here is already a pointer, so need to change anything
template<typename T>
class device_ptr {
private:
	using native_handle_variant = std::variant<T, cl_mem>;
	native_handle_variant handle;

public:
	explicit device_ptr(T cuda_ptr) : handle(cuda_ptr) {}

	explicit device_ptr(cl_mem cl_ptr) : handle(cl_ptr) {}

	template<backend B>
	native_handle_t<T, B> get() const {
		using HandleType = native_handle_t<T, B>;

		if (std::holds_alternative<HandleType>(handle)) {
			return std::get<HandleType>(handle);
		}
		// happens only when you request data for the incorrect backend
		throw std::runtime_error(
			"Mismatched backend: device_ptr does not hold the requested handle type.");
	}
	device_ptr(device_ptr&& other)				   = default;
	device_ptr& operator=(device_ptr&& other)	   = default;
	device_ptr(const device_ptr& other)			   = default;
	device_ptr& operator=(const device_ptr& other) = default;
};
} // namespace raw::device_types

