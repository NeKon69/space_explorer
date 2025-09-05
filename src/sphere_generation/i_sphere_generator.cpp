//
// Created by progamers on 9/5/25.
//
#include "sphere_generation/i_sphere_generator.h"
namespace raw::sphere_generation {
void i_sphere_generator::sync() {
	if (worker_thread.joinable()) {
		worker_thread.join();
	}
}

} // namespace raw::sphere_generation