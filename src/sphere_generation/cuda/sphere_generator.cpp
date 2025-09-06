//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/cuda/sphere_generator.h"

#include "device_types/device_ptr.h"
#include "sphere_generation/cuda/icosahedron_data_manager.h"
#include "sphere_generation/cuda/kernel_launcher.h"

namespace raw::sphere_generation::cuda {
// The implementation detail, using the index sequence pattern you like.
template<device_types::backend B, typename SourceTuple, std::size_t... Is>
auto get_native_pointers(const SourceTuple& source, std::index_sequence<Is...>) {
	return std::make_tuple(std::get<Is>(source).template get<B>()...);
}

// The clean, public-facing function.
template<device_types::backend B, typename SourceTuple>
auto retrieve_data(const SourceTuple& source) {
	constexpr auto tuple_size = std::tuple_size_v<std::decay_t<SourceTuple>>;
	return get_native_pointers<B>(source, std::make_index_sequence<tuple_size> {});
}

void sphere_generator::generate(uint32_t steps, cuda::cuda_stream& stream,
								std::shared_ptr<i_sphere_resource_manager> source,
								graphics::graphics_data&				   graphics_data) {
	if (steps > predef::MAX_STEPS) {
		throw std::runtime_error(std::format(
			"[Error] Amount of steps should not exceed maximum, which is {}, while was given {}",
			predef::MAX_STEPS, steps));
	}
	sync();
	worker_thread = std::jthread([&stream, steps, source, &graphics_data] mutable {
		cudaStream_t											local_stream = stream.stream();
		graphics::gl_context_lock<graphics::context_type::TESS> lock(graphics_data);
		auto													context = source->create_context();
		auto native_data = retrieve_data<device_types::backend::CUDA>(source->get_data());
		std::apply(
			launch_tessellation,
			std::tuple_cat(std::move(native_data), std::make_tuple(std::ref(local_stream), steps)));
		stream.sync();
	});
}

} // namespace raw::sphere_generation::cuda