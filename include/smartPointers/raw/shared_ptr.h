//
// Created by progamers on 5/30/25.
//

#ifndef SMARTPOINTERS_SHARED_PTR_H
#define SMARTPOINTERS_SHARED_PTR_H

#include "helper.h"
#include "smart_ptr_base.h"

namespace raw {
template<typename T>
class shared_ptr_base : public smart_ptr_base<T> {
protected:
	hub* hub_ptr = nullptr;
	friend class weak_ptr<T>;

public:
	// Inherit constructors
	using smart_ptr_base<T>::smart_ptr_base;

	shared_ptr_base() noexcept = default;

	inline explicit shared_ptr_base(std::nullptr_t) noexcept
		: smart_ptr_base<T>(nullptr), hub_ptr(nullptr) {}

	shared_ptr_base(const shared_ptr_base& other) noexcept {
		this->ptr = other.ptr;
		hub_ptr	  = other.hub_ptr;
		if (hub_ptr)
			hub_ptr->increment_use_count();
	}

	shared_ptr_base(shared_ptr_base&& other) noexcept {
		this->ptr	  = other.ptr;
		hub_ptr		  = other.hub_ptr;
		other.ptr	  = nullptr;
		other.hub_ptr = nullptr;
	}

	shared_ptr_base& operator=(const shared_ptr_base& other) noexcept {
		if (this != &other) {
			if (hub_ptr) {
				hub_ptr->decrement_use_count();
			}
			this->ptr = other.ptr;
			hub_ptr	  = other.hub_ptr;
			if (hub_ptr) {
				hub_ptr->increment_use_count();
			}
		}
		return *this;
	}

	shared_ptr_base& operator=(shared_ptr_base&& other) noexcept {
		if (this != &other) {
			if (hub_ptr) {
				hub_ptr->decrement_use_count();
			}
			this->ptr	  = other.ptr;
			hub_ptr		  = other.hub_ptr;
			other.ptr	  = nullptr;
			other.hub_ptr = nullptr;
		}
		return *this;
	}

	void swap(shared_ptr_base& other) noexcept {
		std::swap(this->ptr, other.ptr);
		std::swap(hub_ptr, other.hub_ptr);
	}

	[[nodiscard]] inline size_t use_count() const noexcept {
		return hub_ptr ? hub_ptr->get_use_count() : 0;
	}

	[[nodiscard]] inline bool unique() const noexcept {
		return use_count() == 1;
	}

	~shared_ptr_base() noexcept {
		if (hub_ptr) {
			hub_ptr->decrement_use_count();
		}
#ifdef RAW_SMART_PTR_DEBUG
		std::cout << "Decrementing counter" << std::endl;
		std::cout << "shared_ptr destructor called" << std::endl;
		std::cout << "Pointer address: " << this->ptr << std::endl;
		std::cout << "Use count: " << use_count() << std::endl;
#endif
	}
};

template<typename T>
class shared_ptr : public shared_ptr_base<T> {
public:
	// Inherit constructors
	using shared_ptr_base<T>::shared_ptr_base;

	explicit shared_ptr(weak_ptr<T>& weak) noexcept {
		if (weak.hub_ptr && weak.hub_ptr->try_increment_use_count_if_not_zero()) {
			this->ptr	  = weak.ptr;
			this->hub_ptr = weak.hub_ptr;
		} else {
			this->ptr	  = nullptr;
			this->hub_ptr = nullptr;
		}
	}

	explicit shared_ptr(T* p) noexcept {
		if (p) {
			this->ptr	  = p;
			this->hub_ptr = new hub(this->ptr, nullptr, &raw::delete_single_object<T>,
									&raw::deallocate_hub_for_new_single);
		} else {
			this->ptr	  = nullptr;
			this->hub_ptr = nullptr;
		}
	}

	inline explicit shared_ptr(T* p, hub* hub) noexcept {
		this->ptr	  = p;
		this->hub_ptr = hub;
	}

	inline explicit shared_ptr(unique_ptr<T>&& unique) noexcept {
		this->ptr	  = unique.release();
		this->hub_ptr = new hub(this->ptr, nullptr, &raw::delete_single_object<T>,
								&raw::deallocate_hub_for_new_single);
	}

	shared_ptr& operator=(unique_ptr<T>&& unique) noexcept {
		shared_ptr temp(std::move(unique));
		this->swap(temp);
		return *this;
	}

	shared_ptr& operator=(std::nullptr_t) noexcept {
		if (this->hub_ptr) {
			this->hub_ptr->decrement_use_count();
			this->hub_ptr = nullptr;
		}
		this->ptr = nullptr;
		return *this;
	}

	shared_ptr& operator=(T* ptr) noexcept {
		shared_ptr temp(std::move(ptr));
		this->swap(temp);
		return *this;
	}

	inline void reset(T* p = nullptr) noexcept {
		shared_ptr<T> temp(p);
		this->swap(temp);
	}
};

template<typename T>
class shared_ptr<T[]> : public shared_ptr_base<T[]> {
public:
	// Inherit constructors
	using shared_ptr_base<T[]>::shared_ptr_base;

	explicit shared_ptr(T* p) noexcept {
		if (p != nullptr) {
			this->ptr	  = p;
			this->hub_ptr = new hub(this->ptr, nullptr, &raw::delete_array_object<T>,
									&raw::deallocate_hub_for_new_array);
		} else {
			this->ptr	  = nullptr;
			this->hub_ptr = nullptr;
		}
	}

	inline explicit shared_ptr(weak_ptr<T[]>& weak) noexcept {
		*this = weak.lock();
	}

	inline explicit shared_ptr(unique_ptr<T[]>&& unique) noexcept {
		this->ptr	  = unique.release();
		this->hub_ptr = new hub(this->ptr, nullptr, &raw::delete_array_object<T>,
								&raw::deallocate_hub_for_new_array);
	}

	inline explicit shared_ptr(T* p, hub* hub) noexcept {
		this->ptr	  = p;
		this->hub_ptr = hub;
	}

	shared_ptr& operator=(unique_ptr<T[]>&& unique) noexcept {
		shared_ptr temp(std::move(unique));
		this->swap(temp);
		return *this;
	}

	shared_ptr& operator=(std::nullptr_t) noexcept {
		if (this->hub_ptr) {
			this->hub_ptr->decrement_use_count();
			this->hub_ptr = nullptr;
		}
		this->ptr = nullptr;
		return *this;
	}

	shared_ptr& operator=(T* ptr) noexcept {
		shared_ptr temp(std::move(ptr));
		this->swap(temp);
		return *this;
	}

	inline void reset(T* p = nullptr) noexcept {
		shared_ptr<T[]> temp(p);
		this->swap(temp);
	}
};
} // namespace raw

#endif // SMARTPOINTERS_SHARED_PTR_H
