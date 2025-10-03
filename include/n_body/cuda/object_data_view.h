//
// Created by progamers on 10/3/25.
//

#pragma once
#include <glm/glm.hpp>

#include "device_types/cuda/fwd.h"
#include "n_body/physics_component.h"

namespace raw::n_body::cuda {
template<typename T>
struct object_data_view {
	glm::vec<3, T>* positions  = nullptr;
	glm::vec<3, T>* velocities = nullptr;
	T*				masses	   = nullptr;
	T*				radii	   = nullptr;
	object_data_view()		   = delete;
	explicit object_data_view(soa_device_data<T>& data)
		: positions(data.positions.template get<device_types::backend::CUDA>()),
		  velocities(data.velocities.template get<device_types::backend::CUDA>()),
		  masses(data.masses.template get<device_types::backend::CUDA>()),
		  radii(data.radii.template get<device_types::backend::CUDA>()) {}
};
} // namespace raw::n_body::cuda