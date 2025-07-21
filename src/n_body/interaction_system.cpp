//
// Created by progamers on 7/20/25.
//
#include "n_body/interaction_system.h"

#include <functional>
namespace raw {
interaction_system::interaction_system()
	: d_objects_first(cuda_buffer<space_object>::create(1)),
	  d_objects_second(cuda_buffer<space_object>::create(1)),
	  c_objects(0),
	  threads_to_launch(0),
	  data_changed(false) {}
interaction_system::interaction_system(size_t number_of_planets)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects_first(cuda_buffer<space_object>::create(1)),
	  d_objects_second(cuda_buffer<space_object>::create(1)),
	  c_objects(number_of_planets),
	  threads_to_launch(number_of_planets),
	  data_changed(true) {
	update_threads();
	update_data();
}
interaction_system::interaction_system(const std::vector<space_object>& objects)
	// We'll allocate only one bit, since it'll be reallocated later anyway
	: d_objects_first(cuda_buffer<space_object>::create(1)),
	  d_objects_second(cuda_buffer<space_object>::create(1)),
	  c_objects(objects),

	  threads_to_launch(objects.size()),
	  data_changed(true) {
	update_data();
}
interaction_system::interaction_system(const raw::interaction_system& sys)
	: d_objects_first(sys.d_objects_first),
	  d_objects_second(sys.d_objects_second),
	  c_objects(sys.c_objects),
	  threads_to_launch(c_objects.size()),
	  data_changed(false) {}
void interaction_system::update_data() {
	if (data_changed) {
		d_objects_first.allocate(c_objects.size() * sizeof(space_object));
		d_objects_first.set_data(c_objects.data(), c_objects.size() * sizeof(space_object));
		d_objects_second.allocate(c_objects.size() * sizeof(space_object));
		d_objects_second.set_data(c_objects.data(), c_objects.size() * sizeof(space_object));
		data_changed = false;
	}
}
void interaction_system::update_threads() {
	threads_to_launch = static_cast<unsigned int>(c_objects.size());
}
std::optional<raw::space_object> interaction_system::get()  {
    if(num_of_obj >= c_objects.size()) {
        return std::nullopt;
    }
    return c_objects[num_of_obj++];
}
void interaction_system::update_sim() {
	constexpr auto update_time		   = time(1 * 1000 * 1000);
	auto		   time_since_last_upd = clock.get_elapsed_time();
	time_since_last_upd.to_milli();
	if (time_since_last_upd > update_time) {
		for (UI i = 0; i < c_objects.size(); ++i) {
			// Launch cuda kernels  from different threads (can't use jthread here, since we don't
			// need it to be joined)
			std::thread thread([this, time_since_last_upd, i]() {
				if (number_of_sim % 2 == 0) {
					space_object(c_objects[i])
						.update_position(this->get_first_ptr(), this->get_second_ptr(),
										 time_since_last_upd, c_objects.size());
				} else {
					space_object(c_objects[i])
						.update_position(this->get_second_ptr(), this->get_first_ptr(),
										 time_since_last_upd, c_objects.size());
				}
			});
			thread.detach();
		}
		number_of_sim++;
		clock.restart();
		cudaDeviceSynchronize();
		if (number_of_sim % 2 != 0) {
			cudaMemcpyAsync(c_objects.data(), d_objects_second.get(),
							c_objects.size() * sizeof(space_object), cudaMemcpyDeviceToHost);
		} else {
			cudaMemcpyAsync(c_objects.data(), d_objects_first.get(),
							c_objects.size() * sizeof(space_object), cudaMemcpyDeviceToHost);
		}
	}
}
} // namespace raw