//
// Created by progamers on 9/7/25.
//

#ifndef SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
#define SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
#include <glad/glad.h>
#include <raw_memory.h>

#include "device_types/cuda/from_gl/buffer.h"
#include "graphics/instanced_data.h"
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
	std::vector<physics::space_object_data<T>>					  staging_buffer;
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
		int obj_size		= sizeof(graphics::instanced_data);
		int single_obj_size = sizeof(glm::vec4);
		glBufferData(GL_ARRAY_BUFFER, obj_size * maximum_amount_of_objects, vec.data(),
					 GL_DYNAMIC_DRAW);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size, nullptr);
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void *)(single_obj_size));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void *)(single_obj_size * 2));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void *)(single_obj_size * 3));
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

	void add(physics::space_object_data<T> object) {
		staging_buffer.push_back(object);
	}
	n_body_data<T> get_data() override {
		if (!staging_buffer.empty()) {
			if (objects.size() + staging_buffer.size() > capacity) {
				throw std::runtime_error("Exceeded maximum number of objects!");
			}
			physics_data.memcpy(staging_buffer.data(), staging_buffer.size(), objects.size(),
								cudaMemcpyHostToDevice);
			objects.insert(objects.end(), staging_buffer.begin(), staging_buffer.end());
			staging_buffer.clear();
		}
		return std::make_tuple(device_ptr(instance_data.get_data()),
							   device_ptr(physics_data.get()), objects.size());
	}
	void remove(size_t index) {
		size_t last_index = objects.size() - 1;
		if (index < last_index) {
			objects[index] = objects[last_index];
			physics_data.memcpy(&objects[index], sizeof(physics::space_object_data<T>), index,
								cudaMemcpyHostToDevice);
		}
		objects.pop_back();
	}
	void update(size_t index, physics::space_object_data<T> object) {
		if (index < objects.size()) {
			objects[index] = object;
			physics_data.memcpy(&objects[index], sizeof(physics::space_object_data<T>), index,
								cudaMemcpyHostToDevice);
		}
	}
	std::vector<physics::space_object_data<T>> get_all_objects() {
		physics_data.memcpy(objects.data(), sizeof(physics::space_object_data<T>) * objects.size(),
							0, cudaMemcpyDeviceToHost);
		return objects;
	}
};
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
