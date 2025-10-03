//
// Created by progamers on 9/7/25.
//

#pragma once
#include "common/fwd.h"
#include "device_types/device_ptr.h"
namespace raw::common {

template<auto B, typename T>
decltype(auto) transform_element(T&& element) {
	if constexpr (requires { std::forward<T>(element).template get<B>(); }) {
		return std::forward<T>(element).template get<B>();
	} else {
		return std::forward<T>(element);
	}
}

template<device_types::backend B, typename SourceTuple, std::size_t... Is>
auto get_native_pointers(SourceTuple&& source, std::index_sequence<Is...>) {
	return std::make_tuple(
		transform_element<B>(std::get<Is>(std::forward<SourceTuple>(source)))...);
}

template<device_types::backend B, typename SourceTuple>
auto retrieve_data(SourceTuple&& source) {
	constexpr auto tuple_size = std::tuple_size_v<std::decay_t<SourceTuple>>;
	return get_native_pointers<B>(std::forward<SourceTuple>(source),
								  std::make_index_sequence<tuple_size> {});
}
} // namespace raw::common
