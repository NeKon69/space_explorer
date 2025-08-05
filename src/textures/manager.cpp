//
// Created by progamers on 8/5/25.
//
#include "textures/manager.h"
namespace raw::texture {

manager::manager(raw::texture::planet_id starting_seed) : current_seed(starting_seed) {}

planet_id manager::get_seed() {
	return current_seed++;
}
} // namespace raw::texture