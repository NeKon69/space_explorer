//
// Created by progamers on 9/18/25.
//

#include "texture_generation/cuda/planet_source.h"
namespace raw::texture_generation::cuda {
void planet_source::prepare_single(texture_handle& handle) {
	auto& array = handle.texture.get();
	handle.texture.map();
	handle.texture_descriptor.set_array(array);
	handle.surface = device_types::cuda::surface(handle.texture_descriptor);
}
void planet_source::cleanup_single(texture_handle& handle) {
	handle.texture.unmap();
}
size_t planet_source::get_pool_index() const {
	return pool_index;
}
void planet_source::set_pool_index(size_t index) {
	pool_index = index;
}

void planet_source::cleanup() {
	cleanup_single(handle_albedo_metal);
	cleanup_single(handle_norm_rough_ao);
}
void planet_source::prepare() {
	prepare_single(handle_albedo_metal);
	prepare_single(handle_norm_rough_ao);
}

std::tuple<cudaSurfaceObject_t, cudaSurfaceObject_t> planet_source::get_surfaces() {
	return std::make_tuple(handle_albedo_metal.surface.get(), handle_norm_rough_ao.surface.get());
}

} // namespace raw::texture_generation::cuda