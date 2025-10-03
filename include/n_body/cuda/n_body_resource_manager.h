//
// Created by progamers on 9/7/25.
//

#pragma once
#include <raw_memory.h>
// clang-format off
#include <unordered_map>


#include "n_body_resource_manager.h"
#include "n_body/i_n_body_resource_manager.h"
#include "device_types/cuda/buffer.h"
#include "device_types/cuda/from_gl/buffer.h"
#include "entity_management/entity_id.h"
#include "graphics/instanced_data.h"
// clang-format on
namespace raw::n_body::cuda {
using namespace raw::device_types;

template<typename T>
struct gpu_objects_data {
	device_types::cuda::buffer<glm::vec<3, T>> positions;
	device_types::cuda::buffer<glm::vec<3, T>> velocities;
	device_types::cuda::buffer<T>			   masses;
	device_types::cuda::buffer<T>			   radii;
	gpu_objects_data(uint32_t amount, std::shared_ptr<device_types::cuda::cuda_stream>& stream)
		: positions(amount * sizeof(glm::vec<3, T>), stream),
		  velocities(amount * sizeof(glm::vec<3, T>), stream),
		  masses(amount * sizeof(T), stream),
		  radii(amount * sizeof(T), stream) {}
};
constexpr int size = sizeof(gpu_objects_data<double>);

template<typename T>
class n_body_resource_manager : public i_n_body_resource_manager<T> {
private:
	std::shared_ptr<device_types::cuda::cuda_stream>			  local_stream;
	device_types::cuda::from_gl::buffer<graphics::instanced_data> instance_data;
	gpu_objects_data<T>											  physics_data_gpu_a;
	gpu_objects_data<T>											  physics_data_gpu_b;
	device_types::cuda::buffer<entity_management::entity_id>	  entity_ids_gpu;

	uint32_t n_body_id = 0;
	size_t	 bytes	   = 0;
	size_t	 capacity  = 0;
	uint32_t vbo	   = 0;

	size_t current_object_amount = 0;

private:
	soa_device_data<T> get_pointers(gpu_objects_data<T>& raii_objects) {
		soa_device_data<T> device_data(
			device_ptr(raii_objects.positions.get()), device_ptr(raii_objects.velocities.get()),
			device_ptr(raii_objects.masses.get()), device_ptr(raii_objects.radii.get()));
		return device_data;
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
							std::vector<space_object_data<T>> starting_objects,
							std::shared_ptr<i_queue>		  stream)
		: local_stream(std::dynamic_pointer_cast<device_types::cuda::cuda_stream>(stream)),
		  physics_data_gpu_a(maximum_objects, local_stream),
		  physics_data_gpu_b(maximum_objects, local_stream),
		  entity_ids_gpu(maximum_objects * sizeof(entity_management::entity_id), local_stream),
		  bytes(bytes),
		  capacity(maximum_objects),
		  vbo(vbo),
		  current_object_amount(starting_objects.size()) {
		if (!local_stream) {
			throw std::invalid_argument(
				"Stream passed was not created or created for another backend!");
		}

		instance_data = device_types::cuda::from_gl::buffer<graphics::instanced_data>(
			&this->bytes, this->vbo, this->local_stream);
		physics_data_gpu_a.memcpy(starting_objects.data(),
								  starting_objects.size() * sizeof(space_object_data<T>), 0,
								  cudaMemcpyHostToDevice);
		physics_data_gpu_b.memcpy(starting_objects.data(),
								  starting_objects.size() * sizeof(space_object_data<T>), 0,
								  cudaMemcpyHostToDevice);
	}

	n_body_context<T> create_context() override {
		return n_body_context<T>(this, vbo);
	}

	n_body_data<T> get_data() override {
		return std::make_tuple(device_ptr(instance_data.get_data()),
							   get_pointers(physics_data_gpu_a), get_pointers(physics_data_gpu_b),
							   current_object_amount);
	}

	void swap_buffers() {
		std::swap(physics_data_gpu_a, physics_data_gpu_b);
	}
	[[nodiscard]] uint32_t get_amount() const override {
		return current_object_amount;
	}
};
} // namespace raw::n_body::cuda
