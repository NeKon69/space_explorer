//
// Created by progamers on 5/27/25.
//

#ifndef SMARTPOINTERS_HELPER_H
#define SMARTPOINTERS_HELPER_H

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>

#include "fwd.h"
#include "hub.h"

// Struct to emulate shared_ptr's internal structure
template<typename T>
struct combined {
	raw::hub hub_ptr;
	T		 ptr;
};

namespace raw {

template<typename T>
static void delete_single_object(void* obj_ptr, size_t) {
	delete static_cast<T*>(obj_ptr);
}

template<typename T>
static void delete_array_object(void* obj_ptr, size_t) {
	using element_type = std::remove_extent_t<T>;
	delete[] static_cast<element_type*>(obj_ptr);
}

template<typename T>
static void destroy_make_shared_object(void* obj_ptr, size_t) {
	static_cast<T*>(obj_ptr)->~T();
}

template<typename T>
static void destroy_make_shared_array(void* obj_ptr, size_t size) {
	using element_type = std::remove_extent_t<T>;

	element_type* array_base_ptr = static_cast<element_type*>(obj_ptr);

	for (size_t i = 0; i < size; ++i) {
		array_base_ptr[i].~element_type();
	}
}

static void deallocate_hub_for_new_single(void* hub_ptr, void*) {
	delete static_cast<hub*>(hub_ptr);
}

static void deallocate_hub_for_new_array(void* hub_ptr, void*) {
	delete static_cast<hub*>(hub_ptr);
}

static void deallocate_make_shared_block(void*, void* base_block_ptr) {
	std::free(base_block_ptr);
}

/**
 * @brief Creates a unique_ptr that manages a static array.
 * @param size size of the array.
 */
template<typename T>
std::enable_if_t<std::is_array_v<T>, raw::unique_ptr<T>> make_unique(size_t size) {
	using element_type = std::remove_extent_t<T>;
	return raw::unique_ptr<T>(new element_type[size]());
}

template<typename T, typename... Args>
/**
 * @brief Creates a unique_ptr that manages a single object.
 * @param args Constructor arguments for the new object.
 */
std::enable_if_t<!std::is_array_v<T>, raw::unique_ptr<T>> make_unique(Args&&... args) {
	return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<typename T, typename... Args>
/**
 * @brief Creates a shared_ptr that manages a single object.
 * @param args Constructor arguments for the new object.
 */
std::enable_if_t<!std::is_array_v<T>, raw::shared_ptr<T>> make_shared(Args&&... args) {
	// Allocate a block of memory that can hold both the object and the hub
	std::byte* raw_block =
		static_cast<std::byte*>(std::aligned_alloc(alignof(combined<T>), sizeof(combined<T>)));
	if (!raw_block) {
		throw std::bad_alloc();
	}
	T*	 constructed_ptr = nullptr;
	hub* constructed_hub = nullptr;
	try {
		// Try constructing the object and the hub in the allocated memory with proper alignment
		constructed_hub = new (raw_block + offsetof(combined<T>, hub_ptr))
			hub(constructed_ptr, raw_block, &destroy_make_shared_object<T>,
				deallocate_make_shared_block);
		constructed_ptr =
			new (raw_block + offsetof(combined<T>, ptr)) T(std::forward<Args>(args)...);
		constructed_hub->set_managed_object_ptr(constructed_ptr);
	} catch (const std::exception& e) {
		// If construction fails, clean up constructed object and then free the memory
		if (constructed_ptr != nullptr) {
			constructed_ptr->~T();
		}
		std::free(raw_block);
		throw;
	}

	return shared_ptr<T>(constructed_ptr, constructed_hub);
}

/**
 * @brief Creates a shared_ptr that manages a static array.
 * @param size size of the array.
 */
template<typename T>
std::enable_if_t<std::is_array_v<T>, raw::shared_ptr<T>> make_shared(size_t size) {
	using element_type = std::remove_extent_t<T>;

	size_t hub_align		 = alignof(raw::hub);
	size_t data_align		 = alignof(element_type);
	size_t overall_alignment = std::max(hub_align, data_align);

	size_t hub_size	   = sizeof(raw::hub);
	size_t data_offset = (hub_size + overall_alignment - 1) / overall_alignment * overall_alignment;
	size_t total_block_size			  = data_offset + size * sizeof(element_type);
	size_t unaligned_total_block_size = data_offset + size * sizeof(element_type);
	size_t aligned_total_block_size	  = (unaligned_total_block_size + overall_alignment - 1) /
									  overall_alignment * overall_alignment;

	std::byte* raw_block =
		static_cast<std::byte*>(std::aligned_alloc(overall_alignment, aligned_total_block_size));
	if (!raw_block) {
		throw std::bad_alloc();
	}

	element_type* constructed_ptr = nullptr;
	raw::hub*	  constructed_hub = nullptr;

	try {
		constructed_hub =
			new (raw_block) raw::hub(nullptr, raw_block, &destroy_make_shared_array<element_type>,
									 &deallocate_make_shared_block, size);

		constructed_ptr = new (raw_block + data_offset) element_type[size]();
		constructed_hub->set_managed_object_ptr(constructed_ptr);

	} catch (...) {
		if (constructed_ptr != nullptr) {
			for (size_t i = 0; i < size; ++i) {
				constructed_ptr[i].~element_type();
			}
		}
		if (constructed_hub != nullptr) {
			constructed_hub->~hub();
		}
		std::free(raw_block);
		throw;
	}

	return raw::shared_ptr<T>(constructed_ptr, constructed_hub);
}

} // namespace raw

#endif // SMARTPOINTERS_HELPER_H