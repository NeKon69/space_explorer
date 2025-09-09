//
// Created by progamers on 9/7/25.
//

#pragma once
#include <raw_memory.h>
// clang-format off
#include "n_body/i_n_body_resource_manager.h"
#include "device_types/cuda/buffer.h"
#include "device_types/cuda/from_gl/buffer.h"
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
	device_types::cuda::buffer<space_object_data<T>>			  physics_data;
	std::vector<space_object_data<T>>							  objects;
	std::vector<pending_action<space_object_data<T>>>			  pending_actions;

	uint32_t n_body_id = 0;
	size_t	 bytes	   = 0;
	size_t	 capacity  = 0;
	uint32_t vbo	   = 0;

private:
	void update_data();
	void init(std::vector<space_object_data<T>> objects) {
		this->objects = objects;
		instance_data = device_types::cuda::from_gl::buffer<graphics::instanced_data>(&bytes, vbo,
																					  local_stream);
		device_types::cuda::resource::mapped_resource guard = instance_data.get_resource();

		physics_data = device_types::cuda::buffer<space_object_data<T>>(bytes, local_stream);
		if (!objects.empty()) {
			physics_data.memcpy(objects.data(), objects.size(), 0, cudaMemcpyHostToDevice);
		}
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
							std::shared_ptr<i_queue>		  stream,
							std::vector<space_object_data<T>> initial_objects = nullptr)
		: local_stream(std::dynamic_pointer_cast<device_types::cuda::cuda_stream>(stream)),
		  physics_data(local_stream),
		  vbo(vbo),
		  bytes(bytes),
		  objects(std::move(initial_objects)),
		  capacity(maximum_objects) {
		if (!local_stream) {
			throw std::invalid_argument(
				"Stream passed was not created or created for another backend!");
		}
		init(objects);
	}

	n_body_context<T> create_context() override {
		update_data();
		return n_body_context<T>(this, vbo);
	}

	uint32_t add(space_object_data<T> object) {
		if (objects.size() + pending_actions.size() >= capacity) {
			std::cerr << "Couldn't add another object!\n";
			return UINT32_MAX;
		}
		object.id = ++n_body_id;
		pending_actions.push_back(pending_action<space_object_data<T>>::add(object));
		return object.id;
	}
	n_body_data<T> get_data() override {
		return std::make_tuple(device_ptr(instance_data.get_data()), device_ptr(physics_data.get()),
							   objects.size());
	}
	void remove(uint32_t id) {
		pending_actions.push_back(pending_action<space_object_data<T>>::remove(id));
	}
	[[nodiscard]] uint32_t get_amount() const override {
		return objects.size();
	}
};
} // namespace raw::n_body::cuda
