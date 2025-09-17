//
// Created by progamers on 9/12/25.
//

#pragma once
#include <glm/vec2.hpp>

#include "deleters/custom_deleters.h"
#include "device_types/cuda/from_gl/image.h"
#include "device_types/cuda/resource_description.h"
#include "device_types/cuda/surface.h"
#include "raw/unique_ptr.h"

namespace raw::texture_generation::cuda {

struct texture_handle {
	// Used to access opengl texture by CUDA
	device_types::cuda::from_gl::image texture;
	// Used to configure surface
	device_types::cuda::resource_description<device_types::cuda::resource_types::array>
		texture_descriptor;
	// Used to write to the texture
	device_types::cuda::surface surface;
};
struct planet_source {
	texture_handle							   handle_albedo_metal;
	texture_handle							   handle_norm_rough_ao;
	unique_ptr<uint32_t, deleters::gl_texture> gl_normal_rough_ao_texture;
	size_t									   pool_index;
};
class planet_resource_manager;
} // namespace raw::texture_generation::cuda