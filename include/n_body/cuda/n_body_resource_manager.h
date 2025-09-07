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
class n_body_resource_manager : i_n_body_resource_manager<T> {
private:
	device_types::cuda::from_gl::buffer<graphics::instanced_data>	  instance_data;
	device_types::cuda::buffer<n_body::physics::space_object_data<T>> physics_data;
	std::vector<n_body::physics::space_object_data<T>>				  objects;
	unique_ptr<uint32_t, deleters::gl_buffer>						  vbo;
	size_t															  bytes = 0;

private:
	void init(uint32_t number_of_attr, uint32_t maximum_amount_of_objects,
			  std::shared_ptr<device_types::cuda::cuda_stream> stream) {
		// this sets up first instanced data (models), textures should be set up somewhere different
		glGenBuffers(1, vbo.get());
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		std::vector<glm::mat4> vec(maximum_amount_of_objects, 1.0f);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * 1000, vec.data(), GL_DYNAMIC_DRAW);

		int obj_size		= sizeof(graphics::instanced_data);
		int single_obj_size = sizeof(glm::vec4);

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

		instance_data =
			device_types::cuda::from_gl::buffer<graphics::instanced_data>(&bytes, *vbo, stream);
		physics_data.allocate(bytes);
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
							std::shared_ptr<device_types::cuda::cuda_stream> stream)
		: physics_data(stream) {
		glBindVertexArray(vao);
		init(number_of_attribute, maximum_objects, stream);
	}

	n_body_context<T> create_context() override {
		return n_body_context<T>(this, vbo);
	}

	n_body_data<T> get_data() const override {
		return std::make_tuple(device_ptr(instance_data), device_ptr(physics_data));
	}
};
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_CUDA_N_BODY_RESOURCE_MANAGER_H
