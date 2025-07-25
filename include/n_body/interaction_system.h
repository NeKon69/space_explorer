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
inline void print_mat_ptr(raw::unique_ptr<glm::mat4[]> gg) {
	for (int i = 0; i < 1; ++i) {
		std::cout << "\t\tBEGINNING OF MATRIX " << i;
		std::cout << "\n\t\t";
		for (int i_mat = 0; i_mat < 4; ++i_mat) {
			for (int j_mat = 0; j_mat < 4; ++j_mat) {
				std::cout << gg[i][i_mat][j_mat] << "\t";
			}
			std::cout << "\n\t\t";
		}
		std::cout << "END OF MATRIX\n\n";
	}
}
template<typename T>
inline void print_vec(glm::vec<3, T> vec) {
	for (int i = 0; i < 3; ++i) {
		std::cout << vec[i] << "\t";
	}
}

template<typename T>
inline void print_space_data(raw::unique_ptr<space_object<T>[]> gg) {
	for (int i = 0; i < 1; ++i) {
		std::cout << "\t\t BEGINNING OF OBJECT DATA\n";
		auto gamers = gg[i].get();
		std::cout << "\t\tPOSITION - ";
		print_vec(gamers.position);
        std::cout << "\n\n";
		std::cout << "\t\tVELOCITY - ";
		print_vec(gamers.velocity);
        std::cout << "\n\n";
		std::cout << "\t";
		std::cout << "END OF OBJECT DATA\n\n";
	}
}

// BEFORE CONSTRUCTING ANYTHING REMEMBER TO BIND VAO
template<typename T>
class interaction_system {
private:
	cuda_buffer<space_object<T>>				 d_objects;
	std::vector<space_object<T>>				 c_objects;
	size_t										 amount_of_bytes = 0;
	cuda_from_gl_data<glm::mat4>				 d_objects_model;
	bool										 data_changed;
	bool										 paused		   = false;
	unsigned int								 number_of_sim = 0;
	unsigned int								 num_of_obj	   = 0;
	raw::clock									 clock;
	raw::unique_ptr<raw::UI, deleter::gl_buffer> vbo;
	friend class space_object<>;

	void update_data() {
		if (data_changed) {
			d_objects.allocate(c_objects.size() * sizeof(space_object<T>));
			d_objects.set_data(c_objects.data(), c_objects.size() * sizeof(space_object<T>));
		}
		data_changed = false;
	}

	void setup(UI number_of_attr) {
		glGenBuffers(1, vbo.get());
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * 1000, nullptr, GL_DYNAMIC_DRAW);

		int obj_size = sizeof(glm::mat4);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size, nullptr);
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(obj_size / 4));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(obj_size / 2));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		glVertexAttribPointer(number_of_attr, 4, GL_FLOAT, GL_FALSE, obj_size,
							  (void*)(obj_size / 4 * 3));
		glEnableVertexAttribArray(number_of_attr);
		glVertexAttribDivisor(number_of_attr++, 1);

		{
			auto err = glGetError();
			std::cout << err << "\n";
		}

		cuda_from_gl_data<glm::mat4> gg(&amount_of_bytes, *vbo);

		d_objects_model = std::move(gg);
		d_objects_model.unmap();
	}

public:
	explicit interaction_system(UI number_of_attr = 2)
		: d_objects(cuda_buffer<space_object<T>>::create(sizeof(space_object<T>))),
		  c_objects(1),
		  data_changed(false),
		  vbo(new UI(0)) {
		setup(number_of_attr);
	}
	explicit interaction_system(size_t number_of_planets, UI number_of_attr = 2)
		// We'll allocate only one bit, since it'll be reallocated later anyway
		: d_objects(cuda_buffer<space_object<T>>::create(1)),
		  c_objects(number_of_planets),
		  data_changed(true),
		  vbo(new UI(0)) {
		update_data();
		clock.restart();
	}
	explicit interaction_system(const std::vector<space_object<T>>& objects, UI number_of_attr = 2)
		// We'll allocate only one bit, since it'll be reallocated later anyway (but we do that so
		// we can have the same stream for all data)
		: d_objects(cuda_buffer<space_object<T>>::create(0)),
		  c_objects(objects),

		  data_changed(true),
		  vbo(new UI(0)) {
		setup(number_of_attr);
		update_data();
		clock.restart();
	}
	interaction_system(interaction_system&& sys, UI number_of_attr = 2) noexcept
		: d_objects(std::move(sys.d_objects)),
		  c_objects(std::move(sys.c_objects)),
		  data_changed(false),
		  vbo(std::move(sys.vbo)) {
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

	void add(const space_object<T>& object) {
		c_objects.push_back(object);
		data_changed = true;
		update_data();
	}

	void setup_model(UI model_vbo) {
		d_objects_model = cuda_from_gl_data<glm::mat4>(&amount_of_bytes, model_vbo);
		d_objects_model.unmap();
	}

	UI get_vbo() const {
		return *vbo;
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

			raw::unique_ptr<space_object<T>[]> ptr_data(make_unique<space_object<T>[]>(5));
			cudaMemcpy(ptr_data.get(), d_objects.get(), 5 * sizeof(space_object<T>),
					   cudaMemcpyDeviceToHost);
            print_space_data(std::move(ptr_data));

			// FIXME: add this thing to another kernel just for fun cause rn it's not really
			// working as i switched to instancing
			//			static auto ke_total = 0.0;
			//			auto		e_kin	 = 0.0;
			//			auto		e_pot	 = 0.0;
			//			for (int i = 0; i < c_objects.size(); ++i) {
			//				auto curr_vel = c_objects[i].get().velocity;
			//				e_kin += 0.5 * c_objects[i].get().mass * curr_vel.x * curr_vel.x +
			//						 curr_vel.y * curr_vel.y + curr_vel.z * curr_vel.z;
			//			}
			//			for (int i = 0; i < c_objects.size(); ++i) {
			//				for (auto j = i; j < c_objects.size(); ++j) {
			//					if (j == i)
			//						continue;
			//					e_pot +=
			//						-predef::G * c_objects[i].get().mass *
			// c_objects[j].get().mass /
			// glm::distance(c_objects[i].get().position,
			// c_objects[j].get().position);
			//				}
			//			}
			//			static auto prev_ke_sum = 0.0;
			//			static auto number		= 0;
			//			++number;
			//			prev_ke_sum += ke_total;
			//			ke_total = e_kin + e_pot;
			//			if (glm::distance(ke_total, prev_ke_sum / number) > 0.1) {
			//				// Lol i am so good, almost never reached this (it's reached only
			// when
			// some really
			//				// wierd stuff happening and epsilon gets to work a lot with a lot
			// of
			// planets at
			//				// once (for dummies - that means the objects are too close to each
			// other)) That
			//				// means I didn't fucked up any physics, right? Oh and btw, I get
			// new
			// interaction of
			//				// objects each time I launch the app, I think it's normal since
			// they
			// don't call
			//				// that sim "n-body problem" a problem for nothing.
			//				std::cerr << "I suck at this.\n";
			//			}
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
	return gg;
}

} // namespace predef
} // namespace raw

#endif // SPACE_EXPLORER_INTERACTION_SYSTEM_H
