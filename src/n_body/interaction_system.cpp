//
// Created by progamers on 7/20/25.
//
#include "n_body/interaction_system.h"

#include <functional>

#include "n_body_predef.h"
namespace raw {
interaction_system::interaction_system()
	: d_objects_first(cuda_buffer<space_object>::create(1)),
	  c_objects(0),
	  threads_to_launch(0),
	  data_changed(false) {}
interaction_system::interaction_system(size_t number_of_planets)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects_first(cuda_buffer<space_object>::create(1)),
	  c_objects(number_of_planets),
	  data_changed(true) {
	update_data();
	clock.restart();
}
interaction_system::interaction_system(const std::vector<space_object>& objects)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects_first(cuda_buffer<space_object>::create(0)),
	  c_objects(objects),

	  data_changed(true) {
	update_data();
	clock.restart();
}
interaction_system::interaction_system(const raw::interaction_system& sys)
	: d_objects_first(sys.d_objects_first), c_objects(sys.c_objects), data_changed(false) {
	clock.restart();
}
void interaction_system::update_data() {
	if (data_changed) {
		d_objects_first.allocate(c_objects.size() * sizeof(space_object));
		d_objects_first.set_data(c_objects.data(), c_objects.size() * sizeof(space_object));
		data_changed = false;
	}
}
std::optional<raw::space_object> interaction_system::get() {
	if (num_of_obj >= c_objects.size()) {
		num_of_obj = 0;
		return std::nullopt;
	}
	return c_objects[num_of_obj++];
}

space_object& interaction_system::operator[](size_t index) {
	return c_objects[index];
}

const space_object& interaction_system::operator[](size_t index) const {
	return c_objects[index];
}
void interaction_system::update_sim() {
	constexpr auto update_time		   = time(10);
	auto		   time_since_last_upd = clock.get_elapsed_time();
	time_since_last_upd.to_milli();
	if (time_since_last_upd > update_time) {
		space_object::update_position(this->get_first_ptr(), time_since_last_upd, c_objects.size());
		number_of_sim++;
		clock.restart();
		cudaDeviceSynchronize();
		cudaMemcpyAsync(c_objects.data(), d_objects_first.get(),
						c_objects.size() * sizeof(space_object), cudaMemcpyDeviceToHost);
		cudaStreamSynchronize(nullptr);
		//		for (const auto& obj : c_objects)
		//			std::cout << "[Debug] Pos: {" << obj.object_data.position.x << ", "
		//					  << obj.object_data.position.y << ", " << obj.object_data.position.z <<
		//"}\n";
		//
		// FIXME: add this thing to another kernel just for fun
		static auto ke_total = 0.0;
		auto		e_kin	 = 0.0;
		auto		e_pot	 = 0.0;
		for (int i = 0; i < c_objects.size(); ++i) {
			auto curr_vel = c_objects[i].get().velocity;
			e_kin += 0.5 * c_objects[i].get().mass * curr_vel.x * curr_vel.x +
					 curr_vel.y * curr_vel.y + curr_vel.z * curr_vel.z;
		}
		for (int i = 0; i < c_objects.size(); ++i) {
			for (auto j = i; j < c_objects.size(); ++j) {
				if (j == i)
					continue;
				e_pot += -predef::G * c_objects[i].get().mass * c_objects[j].get().mass /
						 glm::distance(c_objects[i].get().position, c_objects[j].get().position);
			}
		}
		auto prev_ke = ke_total;
		ke_total	 = e_kin + e_pot;
		if (glm::distance(ke_total, prev_ke) > 0.01) {
			// Lol i am so good, almost never reached this (it's reached only when some really wierd
			// stuff happening and epsilon gets to work a lot with a lot of planets at once (for
			// dummies means the objects are too close to each other))
			// That means I didn't fucked up any physics, right?
			// Oh and btw, I get new interaction of objects each time I launch the app, I think it's
			// normal since they don't call that sim "n-body problem" a problem for nothing.
			std::cerr << "I suck at this.\n";
		}
	}
}

} // namespace raw