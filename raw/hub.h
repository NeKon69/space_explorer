//
// Created by progamers on 5/27/25.
//

#ifndef SMARTPOINTERS_HUB_H
#define SMARTPOINTERS_HUB_H

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "fwd.h"

namespace raw {
class hub {
public:
#ifdef RAW_MULTI_THREADED
	std::atomic<size_t> use_count;
	std::atomic<size_t> weak_count;
#else
	size_t use_count;
	size_t weak_count;
#endif

	void*	   managed_object_ptr;
	std::byte* allocated_base_block;

	void (*destroy_obj_func)(void*, size_t);
	void (*deallocate_mem_func)(void*, void*);

	size_t obj_size;

	// Конструктор hub'а
	hub(void* obj_ptr, std::byte*				  base_block, void (*destroyer)(void*, size_t),
		void (*deallocator)(void*, void*), size_t size = 0) noexcept
		: use_count(1),
		  weak_count(0),
		  managed_object_ptr(obj_ptr),
		  allocated_base_block(base_block),
		  destroy_obj_func(destroyer),
		  deallocate_mem_func(deallocator),
		  obj_size(size) {}
	~hub() = default;

	inline void increment_use_count() noexcept {
#ifdef RAW_MULTI_THREADED
		use_count.fetch_add(1, std::memory_order_relaxed);
#else
		use_count++;
#endif
	}
	inline void decrement_use_count() noexcept {
#ifdef RAW_MULTI_THREADED
		if (use_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
#else
		if (--use_count == 0) {
#endif
			if (destroy_obj_func) {
				destroy_obj_func(managed_object_ptr, obj_size);
				managed_object_ptr = nullptr;
			}
#ifdef RAW_MULTI_THREADED
			if (weak_count.load(std::memory_order_acquire) == 0) {
#else
			if (weak_count == 0) {
#endif
				if (deallocate_mem_func) {
					deallocate_mem_func(this, allocated_base_block);
				}
			}
		}
	}

	inline bool try_increment_use_count_if_not_zero() {
#ifdef RAW_MULTI_THREADED
		size_t current_count = use_count.load(std::memory_order_relaxed);
		while (current_count > 0) {
			if (use_count.compare_exchange_weak(current_count, current_count + 1,
												std::memory_order_acquire,
												std::memory_order_relaxed)) {
				return true;
			}
		}
#else
		if (use_count > 0) {
			use_count++;
			return true;
		}
#endif
		return false;
	}

	inline void increment_weak_count() noexcept {
#ifdef RAW_MULTI_THREADED
		weak_count.fetch_add(1, std::memory_order_relaxed);
#else
		weak_count++;
#endif
	}
	inline void decrement_weak_count() noexcept {
#ifdef RAW_MULTI_THREADED
		if (weak_count.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
			use_count.load(std::memory_order_acquire) == 0) {
#else
		if (--weak_count == 0 && use_count == 0) {
#endif
			if (deallocate_mem_func) {
				deallocate_mem_func(this, allocated_base_block);
			}
		}
	}

	inline void set_managed_object_ptr(void* obj_ptr) noexcept {
		managed_object_ptr = obj_ptr;
	}

	inline size_t get_use_count() const noexcept {
#ifdef RAW_MULTI_THREADED
		return use_count.load(std::memory_order_acquire);
#else
		return use_count;
#endif
	}
};

} // namespace raw

#endif // SMARTPOINTERS_HUB_H
