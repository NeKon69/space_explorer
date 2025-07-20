//
// Created by progamers on 7/20/25.
//
#include "objects/interaction_system.h"

namespace raw {
interaction_system::interaction_system()
	: d_objects(sizeof(space_object)), c_objects(0), data_changed(false) {}
interaction_system::interaction_system(size_t number_of_planets)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects(cuda_buffer<space_object>::create(1)),
	  c_objects(number_of_planets),
	  data_changed(true) {
	update_data();
}
interaction_system::interaction_system(const std::vector<space_object>& objects)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects(cuda_buffer<space_object>::create(1)), c_objects(objects), data_changed(true) {
	update_data();
}
void interaction_system::update_data() {
	if (data_changed) {
		d_objects.allocate(c_objects.size() * sizeof(space_object));
		d_objects.set_data(c_objects.data(), c_objects.size() * sizeof(space_object));
		data_changed = false;
	}
}
void interaction_system::update_sim() {
	constexpr auto update_time		   = time(1 * 1000 * 1000);
	auto		   time_since_last_upd = clock.get_elapsed_time();
	time_since_last_upd.to_milli();
	if (time_since_last_upd > update_time) {
		for (auto obj : c_objects) {
			obj.update_position(*this, time_since_last_upd);
		}
        clock.restart();
	}
}
} // namespace raw