//
// Created by progamers on 7/20/25.
//
#include "objects/drawable_space_object.h"

namespace raw {
drawable_space_object::drawable_space_object(const raw::shared_ptr<raw::shader> &shader,
											 const raw::space_object			&data)
	: space_object(data), sphere(shader, static_cast<float>(data.object_data.radius)) {}
} // namespace raw