//
// Created by progamers on 5/27/25.
//

#pragma once

#include <iostream>

#include "fwd.h"

namespace raw {
    template<typename T>
    class smart_ptr_base {
    protected:
        T *ptr = nullptr;
        friend class unique_ptr<T>;
        friend class shared_ptr<T>;

        static_assert(sizeof(T) != 0, "Provide full type or fuck yourself!");

    public:
        constexpr smart_ptr_base() = default;

        explicit smart_ptr_base(std::nullptr_t) noexcept : ptr(nullptr) {
        }

        explicit smart_ptr_base(T *p) noexcept {
            ptr = p;
        }

        inline explicit operator bool() const noexcept {
            return ptr != nullptr;
        }

        inline T *get() const noexcept {
            return ptr;
        }

        inline T &operator*() const noexcept {
            return *ptr;
        }

        inline T *operator->() const noexcept {
            return ptr;
        }

        inline bool operator==(const smart_ptr_base &other) const noexcept {
            return ptr == other.ptr;
        }

        inline bool operator!=(const smart_ptr_base &other) const noexcept {
            return ptr != other.ptr;
        }

        inline bool operator>(const smart_ptr_base &other) const noexcept {
            return ptr > other.ptr;
        }

        inline bool operator<(const smart_ptr_base &other) const noexcept {
            return ptr < other.ptr;
        }

        inline bool operator>=(const smart_ptr_base &other) const noexcept {
            return ptr >= other.ptr;
        }

        inline bool operator<=(const smart_ptr_base &other) const noexcept {
            return ptr <= other.ptr;
        }

        std::ostream &operator<<(std::ostream &os) const {
            os << (ptr ? "Address=" + std::to_string(reinterpret_cast<uintptr_t>(ptr)) : "Null pointer")
                    << ": " << (!ptr ? "null" : *ptr);
            return os;
        }
    };

    template<typename T>
    class smart_ptr_base<T[]> {
    protected:
        T *ptr = nullptr;
        friend class unique_ptr<T[]>;
        friend class std::shared_ptr<T[]>;

    public:
        constexpr smart_ptr_base() = default;

        explicit smart_ptr_base(std::nullptr_t) noexcept : ptr(nullptr) {
        }

        explicit smart_ptr_base(T *p) noexcept {
            ptr = p;
        }

        inline explicit operator bool() const noexcept {
            return ptr != nullptr;
        }

        inline T *get() const noexcept {
            return ptr;
        }

        inline T &operator[](size_t index) const noexcept {
            return ptr[index];
        }

        inline bool operator==(const smart_ptr_base &other) const noexcept {
            return ptr == other.ptr;
        }

        inline bool operator!=(const smart_ptr_base &other) const noexcept {
            return ptr != other.ptr;
        }

        inline bool operator>(const smart_ptr_base &other) const noexcept {
            return ptr > other.ptr;
        }

        inline bool operator<(const smart_ptr_base &other) const noexcept {
            return ptr < other.ptr;
        }

        inline bool operator>=(const smart_ptr_base &other) const noexcept {
            return ptr >= other.ptr;
        }

        inline bool operator<=(const smart_ptr_base &other) const noexcept {
            return ptr <= other.ptr;
        }
    };
} // namespace raw

