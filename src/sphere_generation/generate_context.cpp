//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/generation_context.h"
#include "sphere_generation/i_sphere_resource_manager.h"

namespace raw::sphere_generation {
generation_context::generation_context(i_sphere_resource_manager* mgr, uint32_t vbo,
									   uint32_t ebo)
	: manager(mgr) {
	manager->prepare(vbo, ebo);
}
generation_context::~generation_context() {
	std::cout << "Clean Up is CALLED!!!\n";
	manager->cleanup();
}

} // namespace raw::sphere_generation