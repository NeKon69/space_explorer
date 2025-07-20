//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_INTERACTION_SYSTEM_H
#define SPACE_EXPLORER_INTERACTION_SYSTEM_H
#include "space_object.h"
#include "clock.h"
namespace raw {
class interaction_system {
private:
	cuda_buffer<space_object> d_objects;
	std::vector<space_object> c_objects;
	bool					  data_changed;
	raw::clock					  clock;
	friend class space_object;

public:
	interaction_system();
	interaction_system(size_t number_of_planets);
	interaction_system(const std::vector<space_object>& objects);
	void update_data();
	void update_sim();
};
} // namespace raw

#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
