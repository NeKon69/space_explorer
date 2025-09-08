//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
#define SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
#include <glad/glad.h>
#include <raw_memory.h>
#include <thrust/remove.h>

#include "device_types/cuda/buffer.h"
#include "device_types/cuda/from_gl/buffer.h"
#include "graphics/instanced_data.h"
#include "n_body/cuda/pending_action.h"
#include "n_body/i_n_body_resource_manager.h"
#include "physics/space_object.h"
namespace raw::n_body::cuda {
using namespace raw::device_types;
template<typename T>
class n_body_resource_manager : public i_n_body_resource_manager<T> {
private:
	device_types::cuda::from_gl::buffer<graphics::instanced_data> instance_data;
	device_types::cuda::buffer<physics::space_object_data<T>>	  physics_data;
	std::vector<physics::space_object_data<T>>					  objects;
	std::shared_ptr<device_types::cuda::cuda_stream>			  local_stream;
	std::vector<pending_action<physics::space_object_data<T>>>	  pending_actions;
	uint32_t													  n_body_id = 0;
	unique_ptr<uint32_t, deleters::gl_buffer>					  vbo;
	size_t														  bytes	   = 0;
	size_t														  capacity = 0;

private:
	void init(std::vector<physics::space_object_data<T>> objects, uint32_t number_of_attr,
			  uint32_t maximum_amount_of_objects) {
		// this sets up first instanced data (models), textures should be set up somewhere different
		glGenBuffers(1, vbo.get());
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		std::vector<graphics::instanced_data> vec(maximum_amount_of_objects, {1.0f, 0, 0});
		if (objects != nullptr) {
			vec = objects;
			vec.resize(maximum_amount_of_objects);
		}
		const int obj_size		  = sizeof(graphics::instanced_data);
		const int single_obj_size = sizeof(glm::vec4);
		glBufferData(GL_ARRAY_BUFFER, obj_size * maximum_amount_of_objects, vec.data(),
					 GL_DYNAMIC_DRAW);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size, nullptr);
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(single_obj_size));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(single_obj_size * 2));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(single_obj_size * 3));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glBindVertexArray(0);
	}

protected:
	void prepare(uint32_t vbo) override {
		instance_data.map();
	}
	void cleanup() override {
		instance_data.unmap();
	}

public:
	n_body_resource_manager(uint32_t vao, uint32_t number_of_attribute, uint32_t maximum_objects,
							std::shared_ptr<i_queue>				   stream,
							std::vector<physics::space_object_data<T>> initial_objects = nullptr)
		: local_stream(std::dynamic_pointer_cast<device_types::cuda::cuda_stream>(stream)),
		  vbo(new uint32_t(0)),
		  objects(std::move(initial_objects)),
		  capacity(maximum_objects) {
		if (!local_stream) {
			throw std::invalid_argument(
				"Stream passed was not created or created for another backend!");
		}
		glBindVertexArray(vao);
		init(objects, number_of_attribute, maximum_objects);
		instance_data = device_types::cuda::from_gl::buffer<graphics::instanced_data>(&bytes, *vbo,
																					  local_stream);
		physics_data =
			device_types::cuda::buffer<physics::space_object_data<T>>(bytes, local_stream);
		if (!objects.empty()) {
			physics_data.memcpy(objects.data(), objects.size(), 0, cudaMemcpyHostToDevice);
		}
	}

	n_body_context<T> create_context() override {
		return n_body_context<T>(this, *vbo);
	}

	uint32_t add(physics::space_object_data<T> object) {
		if (objects.size() + pending_actions.size() >= capacity) {
			std::cerr << "Couldn't add another object!\n";
			return UINT32_MAX;
		}
		object.id = ++n_body_id;
		pending_actions.push_back(pending_action<physics::space_object_data<T>>::add(object));
		return object.id;
	}
	n_body_data<T> get_data() override {
		if (!pending_actions.empty()) {
			std::vector<uint32_t> ids_to_remove;
			for (const auto& action : pending_actions) {
				if (action.type == pending_action_type::REMOVE) {
					ids_to_remove.push_back(action.id_to_remove);
				}
			}

			if (!ids_to_remove.empty()) {
				device_types::cuda::buffer<uint32_t> d_ids_to_remove(ids_to_remove.size(),
																	 local_stream);
				d_ids_to_remove.memcpy(ids_to_remove.data(), ids_to_remove.size(), 0,
									   cudaMemcpyHostToDevice);

				auto new_end_iter = thrust::remove_if(
					thrust::cuda::par.on(local_stream->stream()), physics_data.get(),
					physics_data.get() + objects.size(),
					is_in_set<T> {d_ids_to_remove.get(),
								  d_ids_to_remove.get() + d_ids_to_remove.get_size()});

				size_t new_size = new_end_iter - physics_data.get();
				objects.erase(std::remove_if(objects.begin(), objects.end(),
											 [&](const auto& obj) {
												 return std::find(ids_to_remove.begin(),
																  ids_to_remove.end(),
																  obj.id) != ids_to_remove.end();
											 }),
							  objects.end());

				assert(new_size == objects.size());
			}

			std::vector<physics::space_object_data<T>> objects_to_add;
			for (const auto& action : pending_actions) {
				if (action.type == pending_action_type::ADD) {
					objects_to_add.push_back(action.object_to_add);
				}
			}

			if (!objects_to_add.empty()) {
				physics_data.memcpy(objects_to_add.data(), objects_to_add.size(),
									objects.size() * sizeof(physics::space_object_data<T>),
									cudaMemcpyHostToDevice);
				objects.insert(objects.end(), objects_to_add.begin(), objects_to_add.end());
			}

			pending_actions.clear();
		}

		return std::make_tuple(device_ptr(instance_data.get_data()), device_ptr(physics_data.get()),
							   objects.size());
	}
	void remove(uint32_t id) {
		pending_actions.push_back(pending_action<physics::space_object_data<T>>::remove(id));
	}
};
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
