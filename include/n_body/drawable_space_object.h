//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
#define SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
#include "space_object.h"
namespace raw {
template<typename T>
class drawable_space_object : public space_object<T>, public sphere {
public:
	using space_object<T>::space_object;
	using sphere::sphere;

	drawable_space_object(const raw::shared_ptr<raw::shader> &shader,
						  const raw::space_object<T>		 &data)
		: space_object<T>(data), sphere(shader, static_cast<float>(data.object_data.radius)) {}
	void set_data(space_object<T> &data) {
		space_object<T>::object_data = data.get();
	}
	void update_world_pos() {
		cudaStreamSynchronize(nullptr);
		move(space_object<T>::object_data.position);
		scale(glm::vec3(space_object<T>::object_data.radius));
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
