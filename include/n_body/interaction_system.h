//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_INTERACTION_SYSTEM_H
#define SPACE_EXPLORER_INTERACTION_SYSTEM_H
#include <optional>
#include <thread>

#include "clock.h"
#include "space_object.h"
namespace raw {
class interaction_system {
private:
	cuda_buffer<space_object> d_objects_first;
	std::vector<space_object> c_objects;
	UI						  threads_to_launch;
	bool					  data_changed;
	unsigned int			  number_of_sim = 0;
	unsigned int			  num_of_obj	= 0;
	raw::clock				  clock;
	friend class space_object;

	void update_data();
	void update_threads();

public:
	interaction_system();
	explicit interaction_system(size_t number_of_planets);
	explicit interaction_system(const std::vector<space_object>& objects);
	interaction_system(const interaction_system& sys);
	[[nodiscard]] space_object* get_first_ptr() const {
		// Well, technically, since we dont store anything besides `space_object_data`, this should
		// work fine
		return d_objects_first.get();
	}

	std::optional<space_object> get();
	void						update_sim();
};

namespace predef {
inline auto generate_data_for_sim() {
	return interaction_system(
		{space_object(glm::vec3(0.0f, 0.f, 0.f))});
}

} // namespace predef

} // namespace raw

#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
