//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
#define SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
#include "space_object.h"
namespace raw {
class drawable_space_object : public space_object, public sphere {
public:
	using space_object::space_object;
	using sphere::sphere;

	drawable_space_object(const raw::shared_ptr<raw::shader>& shader, const space_object& data);
	void set_data(space_object& data);
    void update_world_pos();
};
} // namespace raw

#endif // SPACE_EXPLORER_DRAWABLE_SPACE_OBJECT_H
