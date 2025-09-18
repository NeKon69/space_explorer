//
// Created by progamers on 9/18/25.
//

#pragma once

#include "device_types/cuda/from_gl/image.h"
#include "device_types/cuda/resource_description.h"
#include "device_types/cuda/surface.h"
#include "texture_generation/cuda/fwd.h"
#include "texture_generation/fwd.h"

namespace raw::texture_generation::cuda {
struct texture_handle {
	// Used to access opengl texture by CUDA
	device_types::cuda::from_gl::image texture;
	// Used to configure surface
	device_types::cuda::resource_description<device_types::cuda::resource_types::array>
		texture_descriptor;
	// Used to write to the texture
	device_types::cuda::surface surface;
	texture_handle() = default;
};

class planet_source : public i_planet_source {
private:
	texture_handle handle_albedo_metal;
	texture_handle handle_norm_rough_ao;
	size_t		   pool_index;
	friend texture_generation_context;

private:
	static void prepare_single(texture_handle& handle);
	static void cleanup_single(texture_handle& handle);

public:
	planet_source() = default;

	void   prepare() override;
	void   cleanup() override;
	void   set_pool_index(size_t index);
	size_t get_pool_index() const;

	std::tuple<cudaSurfaceObject_t, cudaSurfaceObject_t> get_surfaces() ;
};
} // namespace raw::texture_generation::cuda
#endif // SPACE_EXPLORER_PLANET_SOURCE_H
