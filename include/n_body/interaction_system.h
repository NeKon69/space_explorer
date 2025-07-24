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
template<typename T>
class interaction_system {
private:
	cuda_buffer<space_object<T>> d_objects;
	std::vector<space_object<T>> c_objects;
	size_t						 amount_of_bytes = 0;
	cuda_from_gl_data<glm::mat4> d_objects_model;
	bool						 data_changed;
	bool						 paused;
	unsigned int				 number_of_sim = 0;
	unsigned int				 num_of_obj	   = 0;
	raw::clock					 clock;
    raw::UI vbo;
	friend class space_object<>;

	void update_data() {
		if (data_changed) {
			d_objects.allocate(c_objects.size() * sizeof(space_object<T>));
			d_objects.set_data(c_objects.data(), c_objects.size() * sizeof(space_object<T>));
		}
		data_changed = false;
	}

public:
	interaction_system()
		: d_objects(cuda_buffer<space_object<T>>::create(1)), c_objects(0), data_changed(false) {}
	explicit interaction_system(size_t number_of_planets)
		// We'll allocate only one bit, since it'll be reallocated later anyway
		: d_objects(cuda_buffer<space_object<T>>::create(1)),
		  c_objects(number_of_planets),
		  data_changed(true) {
		update_data();
		clock.restart();
	}
	explicit interaction_system(const std::vector<space_object<T>>& objects)
		// We'll allocate only one bit, since it'll be reallocated later anyway (but we do that so
		// we can have the same stream for all data)
		: d_objects(cuda_buffer<space_object<T>>::create(0)),
		  c_objects(objects),

		  data_changed(true) {
		update_data();
		clock.restart();
	}
	interaction_system(const interaction_system& sys)
		: d_objects(sys.d_objects), c_objects(sys.c_objects), data_changed(false) {
		clock.restart();
	}
	[[nodiscard]] inline space_object<T>* get_first_ptr() const {
		return d_objects.get();
	}

	void pause() {
		paused = true;
		clock.stop();
	}

	void start() {
		paused = false;
		clock.start();
	}

	void setup_model(UI model_vbo) {
		d_objects_model = cuda_from_gl_data<glm::mat4>(&amount_of_bytes, model_vbo);
		d_objects_model.unmap();
        vbo = model_vbo;
	}

    UI get_vbo() const {
        return vbo;
    }

	std::optional<raw::space_object<T>> get() {
		if (num_of_obj >= c_objects.size()) {
			num_of_obj = 0;
			return std::nullopt;
		}
		return c_objects[num_of_obj++];
	}
	const space_object<T>& operator[](size_t index) const {
		return c_objects[index];
	}
	space_object<T>& operator[](size_t index) {
		return c_objects[index];
	}
	[[nodiscard]] inline UI amount() const {
		return c_objects.size();
	}
	void update_sim() {
		if (paused)
			return;
		constexpr auto update_time		   = time(1);
		auto		   time_since_last_upd = clock.get_elapsed_time();
		time_since_last_upd.to_milli();
		if (time_since_last_upd > update_time) {
			d_objects_model.map();
			space_object<T>::update_position(this->get_first_ptr(), d_objects_model.get_data(),
											 time_since_last_upd, c_objects.size());
			number_of_sim++;
			clock.restart();
			cudaDeviceSynchronize();
			//		for (const auto& obj : c_objects)
			//			std::cout << "[Debug] Pos: {" << obj.object_data.position.x << ", "
			//					  << obj.object_data.position.y << ", " <<
			// obj.object_data.position.z <<
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
					e_pot +=
						-predef::G * c_objects[i].get().mass * c_objects[j].get().mass /
						glm::distance(c_objects[i].get().position, c_objects[j].get().position);
				}
			}
			static auto prev_ke_sum = 0.0;
			static auto number		= 0;
			++number;
			prev_ke_sum += ke_total;
			ke_total = e_kin + e_pot;
			if (glm::distance(ke_total, prev_ke_sum / number) > 0.1) {
				// Lol i am so good, almost never reached this (it's reached only when some really
				// wierd stuff happening and epsilon gets to work a lot with a lot of planets at
				// once (for dummies - that means the objects are too close to each other)) That
				// means I didn't fucked up any physics, right? Oh and btw, I get new interaction of
				// objects each time I launch the app, I think it's normal since they don't call
				// that sim "n-body problem" a problem for nothing.
				std::cerr << "I suck at this.\n";
			}
			d_objects_model.unmap();
		}
	}
};

namespace predef {
inline auto generate_data_for_sim() {
	std::initializer_list<space_object<float>> gg = {
		space_object<float>(glm::vec3(0.0f, 0.f, 0.f), predef::BASIC_VELOCITY, 2, sqrt(2)),
		space_object<float>(glm::vec3(25.f)), space_object<float>(glm::vec3(-10.f)),
		space_object<float>(glm::vec3(10, -10, 20), predef::BASIC_VELOCITY, 4, sqrt(0.25))};
	std::vector<space_object<float>> ggg(gg.begin(), gg.end());
	return interaction_system<float>(ggg
									 // First object is some kind of star Lol.
	);
}

} // namespace predef
} // namespace raw

#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
