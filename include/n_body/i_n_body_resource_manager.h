//
// Created by progamers on 9/6/25.
//

#pragma once
#include "device_types/device_ptr.h"
#include "entity_management/entity_manager.h"
#include "graphics/instanced_data.h"
#include "graphics/mesh.h"
#include "n_body/cuda/fwd.h"
#include "n_body/fwd.h"
namespace raw::n_body {
using namespace raw::device_types;
template<typename T>
using n_body_data =
	std::tuple<device_ptr<graphics::instanced_data*>, device_ptr<physics_component<T>*>, uint16_t>;
template<typename T>
using n_body_context = common::scoped_resource_handle<i_n_body_resource_manager<T>>;
template<typename T>
class i_n_body_resource_manager {
protected:
	friend n_body_context<T>;
	virtual void prepare(uint32_t vbo) = 0;
	virtual void cleanup()			   = 0;

public:
	virtual ~i_n_body_resource_manager()			  = default;
	virtual n_body_data<T>		   get_data()		  = 0;
	virtual n_body_context<T>	   create_context()	  = 0;
	[[nodiscard]] virtual uint32_t get_amount() const = 0;
	virtual void sync_data(const std::vector<physics_component<T>>&			cpu_physics_data,
						   const std::vector<entity_management::entity_id>& cpu_entity_ids) = 0;
};
} // namespace raw::n_body
