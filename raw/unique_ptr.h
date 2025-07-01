//
// Created by progamers on 5/28/25.
//

#ifndef SMARTPOINTERS_UNIQUE_PTR_H
#define SMARTPOINTERS_UNIQUE_PTR_H

#include "smart_ptr_base.h"

namespace raw {
template<typename T>
class unique_ptr : public smart_ptr_base<T> {
public:
	// Inherit constructors
	using smart_ptr_base<T>::smart_ptr_base;

	~unique_ptr() noexcept {
#ifdef RAW_SMART_PTR_DEBUG
		std::cout << "Deleting single object" << std::endl;
		std::cout << "unique_ptr destructor called" << std::endl;
		std::cout << "Pointer address: " << this->ptr << std::endl;
#endif
		delete this->ptr;
	}

	// Move constructor
	unique_ptr(unique_ptr&& other) noexcept : smart_ptr_base<T>(std::move(other.ptr)) {
		other.ptr = nullptr;
	}

	// Move assignment operator
	unique_ptr& operator=(unique_ptr&& other) noexcept {
		// Clean up and transfer ownership
		reset(other.release());
		return *this;
	}

	unique_ptr& operator=(std::nullptr_t) noexcept {
		// Clean up the current pointer
		delete this->ptr;
		this->ptr = nullptr;
		return *this;
	}

	// Delete copy constructor and copy assignment operator
	unique_ptr(const unique_ptr&)			 = delete;
	unique_ptr& operator=(const unique_ptr&) = delete;

	T* release() noexcept {
		T* temp	  = this->ptr;
		this->ptr = nullptr;
		return temp;
	}

	void reset(T* p = nullptr) noexcept {
		if (this->ptr == p) {
			return;
		}
		delete this->ptr;
		this->ptr = p;
	}

	void swap(unique_ptr& other) noexcept {
		std::swap(this->ptr, other.ptr);
	}
};

template<typename T>
class unique_ptr<T[]> : public smart_ptr_base<T[]> {
public:
	// Inherit constructors
	using smart_ptr_base<T[]>::smart_ptr_base;

	~unique_ptr() noexcept {
#ifdef RAW_SMART_PTR_DEBUG
		std::cout << "Deleting an array" << std::endl;
		std::cout << "unique_ptr destructor called" << std::endl;
		std::cout << "Pointer address: " << this->ptr << std::endl;
#endif
		delete[] this->ptr;
	}

	// Move constructor
	unique_ptr(unique_ptr&& other) noexcept : smart_ptr_base<T[]>(std::move(other.ptr)) {
		other.ptr = nullptr;
	}

	// Move assignment operator
	unique_ptr& operator=(unique_ptr&& other) noexcept {
		// Clean up and transfer ownership
		reset(other.release());
		return *this;
	}

	unique_ptr& operator=(std::nullptr_t) noexcept {
		// Clean up the current pointer
		delete[] this->ptr;
		this->ptr = nullptr;
		return *this;
	}

	// Delete copy constructor and copy assignment operator
	unique_ptr(const unique_ptr&)			 = delete;
	unique_ptr& operator=(const unique_ptr&) = delete;

	T* release() noexcept {
		T* temp	  = this->ptr;
		this->ptr = nullptr;
		return temp;
	}

	void reset(T* p = nullptr) noexcept {
		if (this->ptr == p) {
			return;
		}
		delete[] this->ptr;
		this->ptr = p;
	}

	void swap(unique_ptr& other) noexcept {
		std::swap(this->ptr, other.ptr);
	}
};
} // namespace raw

#endif // SMARTPOINTERS_UNIQUE_PTR_H
