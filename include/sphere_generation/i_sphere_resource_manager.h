//
// Created by progamers on 9/4/25.
//

#pragma once

#include "cuda/fwd.h"
#include "device_types/device_ptr.h"
#include "graphics/mesh.h"
#include "sphere_generation/fwd.h"
using namespace raw::device_types;
namespace raw::sphere_generation {
using tessellation_data =
	std::tuple<device_ptr<graphics::vertex*>, device_ptr<unsigned*>, device_ptr<cuda::edge*>,
			   device_ptr<graphics::vertex*>, device_ptr<unsigned*>, device_ptr<cuda::edge*>,
			   device_ptr<unsigned*>, device_ptr<unsigned*>, device_ptr<unsigned*>,
			   device_ptr<unsigned*>>;
class i_sphere_resource_manager {
protected:
	friend generation_context;

	virtual void cleanup()							 = 0;
	virtual void prepare(uint32_t vbo, uint32_t ebo) = 0;

public:
	[[nodiscard]] virtual tessellation_data	 get_data() const = 0;
	[[nodiscard]] virtual generation_context create_context() = 0;
	virtual ~i_sphere_resource_manager()					  = default;
};
} // namespace raw::sphere_generation

