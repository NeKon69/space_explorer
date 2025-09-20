//
// Created by progamers on 9/9/25.
//

#pragma once
#include <memory>

#include "graphics/instanced_data_buffer.h"
#include "n_body/cuda/n_body_resource_manager.h"
#include "n_body/cuda/n_body_simulator.h"
#include "n_body/fwd.h"
#include "n_body/interaction_system.h"

namespace raw::n_body {
template<typename T, backend Backend>
struct n_body_factory {
	static std::unique_ptr<interaction_system<T>> create(
		const graphics::instanced_data_buffer& render_buffer,
		std::shared_ptr<device_types::i_queue> gpu_queue, uint32_t max_objects,
		std::vector<physics_component<T>> objects, graphics::graphics_data& graphics)
		requires(Backend == backend::CUDA)
	{
		std::shared_ptr<i_n_body_resource_manager<T>> resource_manager =
			std::make_shared<cuda::n_body_resource_manager<T>>(
				render_buffer.get_vbo(), render_buffer.get_size(), max_objects, gpu_queue);
		std::shared_ptr<i_n_body_simulator<T>> n_body_simulator =
			std::make_shared<cuda::n_body_simulator<T>>();
		return std::make_unique<interaction_system<T>>(resource_manager, n_body_simulator,
													   gpu_queue, objects, graphics);
	}
};
} // namespace raw::n_body
