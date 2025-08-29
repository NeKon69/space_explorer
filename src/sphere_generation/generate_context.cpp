//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/generation_context.h"
#include "sphere_generation/icosahedron_data_manager.h"

namespace raw::sphere_generation {
generation_context::generation_context(icosahedron_data_manager& mgr, UI vbo, UI ebo) : manager(mgr) {
	manager.prepare(vbo, ebo);
}
generation_context::~generation_context() {
	std::cout << "Clean Up is CALLED!!!\n";
	manager.cleanup();
}

} // namespace raw::sphere_generation