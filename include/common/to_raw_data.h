//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_TO_RAW_DATA_H
#define SPACE_EXPLORER_TO_RAW_DATA_H
#include "common/fwd.h"
#include "device_types/device_ptr.h"
namespace raw::common {


template<device_types::backend B, typename SourceTuple, std::size_t... Is>
auto get_native_pointers(const SourceTuple& source, std::index_sequence<Is...>) {
	return std::make_tuple(std::get<Is>(source).template get<B>()...);
}

template<device_types::backend B, typename SourceTuple>
auto retrieve_data(const SourceTuple& source) {
	constexpr auto tuple_size = std::tuple_size_v<std::decay_t<SourceTuple>>;
	return get_native_pointers<B>(source, std::make_index_sequence<tuple_size> {});
}
}
#endif //SPACE_EXPLORER_TO_RAW_DATA_H
