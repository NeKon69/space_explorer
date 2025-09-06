//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/cuda/sphere_generator.h"

#include "sphere_generation/cuda/icosahedron_data_manager.h"
#include "sphere_generation/cuda/kernel_launcher.h"

namespace raw::sphere_generation::cuda {
template<typename TargetTuple, std::size_t... Is>
TargetTuple cast_tuple_impl(const tessellation_data& source, std::index_sequence<Is...>) {
	return std::make_tuple(
		static_cast<std::tuple_element_t<Is, TargetTuple>>(std::get<Is>(source))...);
}

cuda_tessellation_data cast_tuple(const tessellation_data& source) {
	constexpr auto tuple_size = std::tuple_size_v<tessellation_data>;
	return cast_tuple_impl<cuda_tessellation_data>(source, std::make_index_sequence<tuple_size> {});
}
void sphere_generator::generate(uint32_t steps, cuda_types::cuda_stream& stream,
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
		auto data_for_thread = cast_tuple(source->get_data());
		std::apply(launch_tessellation,
				   std::tuple_cat(std::move(data_for_thread),
								  std::make_tuple(std::ref(local_stream), steps)));
		stream.sync();
	});
}

} // namespace raw::sphere_generation::cuda