//
// Created by progamers on 5/31/25.
//

#ifndef SMARTPOINTERS_WEAK_PTR_H
#define SMARTPOINTERS_WEAK_PTR_H

#include "helper.h"
#include "smart_ptr_base.h"

namespace raw {
template<typename T>
class weak_ptr_base : public smart_ptr_base<T> {
protected:
	hub* hub_ptr = nullptr;
	friend class shared_ptr<T>;

public:
	// Inherit constructors
	using smart_ptr_base<T>::smart_ptr_base;

	weak_ptr_base(T* p) = delete;

	weak_ptr_base() noexcept = default;

	weak_ptr_base(std::nullptr_t) noexcept : smart_ptr_base<T>(nullptr), hub_ptr(nullptr) {}

	weak_ptr_base(const weak_ptr_base& other) noexcept {
		this->ptr = other.ptr;
		hub_ptr	  = other.hub_ptr;
		if (hub_ptr)
			hub_ptr->increment_weak_count();
	}

	weak_ptr_base(weak_ptr_base&& other) noexcept {
		this->ptr	  = other.ptr;
		hub_ptr		  = other.hub_ptr;
		other.ptr	  = nullptr;
		other.hub_ptr = nullptr;
	}

	weak_ptr_base& operator=(const weak_ptr_base& other) noexcept {
		if (this != &other) {
			if (hub_ptr) {
				hub_ptr->decrement_weak_count();
			}
			this->ptr = other.ptr;
			hub_ptr	  = other.hub_ptr;
			if (hub_ptr) {
				hub_ptr->increment_weak_count();
			}
		}
		return *this;
	}

	weak_ptr_base& operator=(weak_ptr_base&& other) noexcept {
		if (this != &other) {
			if (hub_ptr) {
				hub_ptr->decrement_weak_count();
			}
			this->ptr	  = other.ptr;
			hub_ptr		  = other.hub_ptr;
			other.ptr	  = nullptr;
			other.hub_ptr = nullptr;
		}
		return *this;
	}

	void reset() noexcept {
		if (hub_ptr) {
			hub_ptr->decrement_weak_count();
			hub_ptr = nullptr;
		}
		this->ptr = nullptr;
	}

	void swap(weak_ptr_base& other) noexcept {
		std::swap(this->ptr, other.ptr);
		std::swap(hub_ptr, other.hub_ptr);
	}

	~weak_ptr_base() noexcept {
		if (hub_ptr) {
			hub_ptr->decrement_weak_count();
		}
#ifdef RAW_SMART_PTR_DEBUG
		std::cout << "Decrementing counter" << std::endl;
		std::cout << "weak_ptr destructor called" << std::endl;
		std::cout << "Pointer address: " << this->ptr << std::endl;
		std::cout << "Use count: " << use_count() << std::endl;
#endif
	}

	inline size_t use_count() const noexcept {
		return (hub_ptr ? hub_ptr->get_use_count() : 0);
	}

	inline bool expired() const noexcept {
#ifdef RAW_SMART_PTR_DEBUG
		std::cout << use_count() << std::endl;
#endif
		return use_count() == 0;
	}

	inline shared_ptr<T> lock() const noexcept {
#ifdef RAW_MULTI_THREADED
		if (this->hub_ptr && this->hub_ptr->try_increment_use_count_if_not_zero()) {
#else
		if (this->hub_ptr && this->hub_ptr->use_count > 0) {
			this->hub_ptr->use_count++;
#endif
			return shared_ptr<T>(this->ptr, this->hub_ptr);
		}
		return shared_ptr<T>();
	}
};
template<typename T>
class weak_ptr : public weak_ptr_base<T> {
public:
	// Inherit constructors
	using weak_ptr_base<T>::weak_ptr_base;

	weak_ptr() noexcept = default;

	weak_ptr(std::nullptr_t) noexcept : weak_ptr_base<T>(nullptr) {}

	weak_ptr(const shared_ptr<T>& shared) noexcept {
		this->ptr	  = shared.get();
		this->hub_ptr = shared.hub_ptr;
		if (this->hub_ptr) {
			this->hub_ptr->increment_weak_count();
		}
	}

	inline weak_ptr& operator=(const shared_ptr<T>& shared) noexcept {
		if (this->hub_ptr) {
			this->hub_ptr->decrement_weak_count();
		}
		this->ptr	  = shared.get();
		this->hub_ptr = shared.hub_ptr;
		if (this->hub_ptr) {
			this->hub_ptr->increment_weak_count();
		}
		return *this;
	}
};
template<typename T>
class weak_ptr<T[]> : public weak_ptr_base<T[]> {
public:
	// Inherit constructors
	using weak_ptr_base<T[]>::weak_ptr_base;

	weak_ptr() noexcept = default;

	weak_ptr(std::nullptr_t) noexcept : weak_ptr_base<T[]>(nullptr) {}

	weak_ptr(const shared_ptr<T[]>& shared) noexcept {
		this->ptr	  = shared.get();
		this->hub_ptr = shared.hub_ptr;
		if (this->hub_ptr) {
			this->hub_ptr->increment_weak_count();
		}
	}

	inline weak_ptr& operator=(const shared_ptr<T[]>& shared) noexcept {
		if (this->hub_ptr) {
			this->hub_ptr->decrement_weak_count();
		}
		this->ptr	  = shared.get();
		this->hub_ptr = shared.hub_ptr;
		if (this->hub_ptr) {
			this->hub_ptr->increment_weak_count();
		}
		return *this;
	}
};

} // namespace raw

#endif // SMARTPOINTERS_WEAK_PTR_H
