//
// Created by progamers on 9/7/25.
//

#pragma once
#include <raw_memory.h>
// clang-format off
#include <c++/15.2.1/unordered_map>


#include "n_body/i_n_body_resource_manager.h"
#include "device_types/cuda/buffer.h"
#include "device_types/cuda/from_gl/buffer.h"
#include "entity_management/entity_id.h"
#include "graphics/instanced_data.h"
#include "n_body/cuda/pending_action.h"
// clang-format on
namespace raw::n_body::cuda {
using namespace raw::device_types;
template<typename T>
class n_body_resource_manager : public i_n_body_resource_manager<T> {
private:
	std::shared_ptr<device_types::cuda::cuda_stream>			  local_stream;
	device_types::cuda::from_gl::buffer<graphics::instanced_data> instance_data;
	device_types::cuda::buffer<physics_component<T>>			  physics_data_gpu;
	device_types::cuda::buffer<entity_management::entity_id>	  entity_ids_gpu;

	uint32_t n_body_id = 0;
	size_t	 bytes	   = 0;
	size_t	 capacity  = 0;
	uint32_t vbo	   = 0;

	size_t current_object_amount = 0;

private:
	template<typename Cont1, typename Cont2>
	bool validate_bounds(const Cont1& cont1, const Cont2& cont2) {
		if (!(cont1.size() > physics_data_gpu.get_size() / sizeof(physics_component<T>) ||
			  cont2.size() > entity_ids_gpu.get_size() / sizeof(entity_management::entity_id))) {
			return false;
		}
		return true;
	}

protected:
	void prepare(uint32_t vbo_) override {
		instance_data.map();
	}
	void cleanup() override {
		instance_data.unmap();
	}

public:
	n_body_resource_manager(uint32_t vbo, size_t bytes, uint32_t maximum_objects,
							std::shared_ptr<i_queue> stream)
		: local_stream(std::dynamic_pointer_cast<device_types::cuda::cuda_stream>(stream)),
		  physics_data_gpu(local_stream),
		  entity_ids_gpu(local_stream),
		  bytes(bytes),
		  capacity(maximum_objects),
		  vbo(vbo) {
		if (!local_stream) {
			throw std::invalid_argument(
				"Stream passed was not created or created for another backend!");
		}
		instance_data = device_types::cuda::from_gl::buffer<graphics::instanced_data>(
			&this->bytes, this->vbo, this->local_stream);
	}

	n_body_context<T> create_context() override {
		return n_body_context<T>(this, vbo);
	}

	n_body_data<T> get_data() override {
		return std::make_tuple(device_ptr(instance_data.get_data()),
							   device_ptr(physics_data_gpu.get()), current_object_amount);
	}
	[[nodiscard]] uint32_t get_amount() const override {
		return current_object_amount;
	}

	void sync_data(const std::vector<physics_component<T>>&			cpu_physics_data,
				   const std::vector<entity_management::entity_id>& cpu_entity_ids) override {
		if (validate_bounds(cpu_physics_data, cpu_entity_ids)) {
			physics_data_gpu.memset(cpu_physics_data.data(),
									cpu_physics_data.size() * sizeof(physics_component<T>),
									cudaMemcpyHostToDevice);
			entity_ids_gpu.memset(cpu_entity_ids.data(),
								  cpu_entity_ids.size() * sizeof(entity_management::entity_id),
								  cudaMemcpyHostToDevice);
		} else {
			std::cerr
				<< "[ERROR] Failed to synchronize properly, falling back to copying maximum available size\n";
			physics_data_gpu.memset(cpu_physics_data.data(), physics_data_gpu.get_size(),
									cudaMemcpyHostToDevice);
			entity_ids_gpu.memset(cpu_entity_ids.data(), entity_ids_gpu.get_size(),
								  cudaMemcpyHostToDevice);
		}
		current_object_amount = cpu_physics_data.get_size();
	}

	std::unordered_map<entity_management::entity_id, uint32_t> build_id_to_index_map() {
		std::vector<entity_management::entity_id>				   host_ids(current_object_amount);
		std::unordered_map<entity_management::entity_id, uint32_t> host_map(current_object_amount);

		CUDA_SAFE_CALL(cudaMemcpyAsync(host_ids.data(), physics_data_gpu.get(),
									   current_object_amount * sizeof(physics_component<T>),
									   cudaMemcpyDeviceToHost));
		for (uint32_t i = 0; i < current_object_amount; i++) {
			host_map[host_ids[i]] = i;
		}

		return host_map;
	}
};
} // namespace raw::n_body::cuda
