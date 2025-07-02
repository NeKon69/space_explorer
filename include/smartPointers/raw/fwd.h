//
// Created by progamers on 5/27/25.
//

#ifndef SMARTPOINTERS_FWD_H
#define SMARTPOINTERS_FWD_H

#include <memory>

namespace raw {

template<typename T>
struct default_deleter;

class hub;

template<typename T>
class smart_ptr_base;

template<typename T, typename D = raw::default_deleter<T>>
class unique_ptr;

template<typename T>
class shared_ptr;

template<typename T>
class weak_ptr;

} // namespace raw

#endif // SMARTPOINTERS_FWD_H
