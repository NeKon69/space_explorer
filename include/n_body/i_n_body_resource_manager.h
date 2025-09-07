//
// Created by progamers on 9/6/25.
//

#ifndef SPACE_EXPLORER_I_N_BODY_DATA_MANAGER_H
#define SPACE_EXPLORER_I_N_BODY_DATA_MANAGER_H
#include <cstdint>
#include <glm/glm.hpp>

#include "device_types/cuda/from_gl/fwd.h"
#include "device_types/device_ptr.h"
#include "graphics/instanced_data.h"
#include "graphics/mesh.h"
#include "n_body/cuda/fwd.h"
#include "n_body/fwd.h"
namespace raw::n_body {
using namespace raw::device_types;
template<typename T>
using n_body_data =
	std::tuple<device_ptr<graphics::instanced_data>, device_ptr<cuda::physics::space_object<T>*>>;
template<typename T>
using n_body_context = common::scoped_resource_handle<i_n_body_resource_manager<T>>;
template<typename T>
class i_n_body_resource_manager {
protected:
	virtual void			  prepare(uint32_t vbo) = 0;
	virtual void			  cleanup()				= 0;

public:
	virtual ~i_n_body_resource_manager()	= default;
	virtual n_body_data<T> get_data() const = 0;
	virtual n_body_context<T> create_context()		= 0;
};
} // namespace raw::n_body
#endif // SPACE_EXPLORER_I_N_BODY_DATA_MANAGER_H
