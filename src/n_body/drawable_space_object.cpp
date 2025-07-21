//
// Created by progamers on 7/20/25.
//
#include "n_body/drawable_space_object.h"

namespace raw {
drawable_space_object::drawable_space_object(const raw::shared_ptr<raw::shader> &shader,
											 const raw::space_object			&data)
	: space_object(data), sphere(shader, static_cast<float>(data.object_data.radius)) {}

void drawable_space_object::set_data(space_object &data) {
	object_data = data.get();
}
void drawable_space_object::update_world_pos() {
	cudaStreamSynchronize(nullptr);
	move(object_data.position);
}
} // namespace raw