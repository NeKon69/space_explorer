//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_SPHERE_GENERATOR_H
#define SPACE_EXPLORER_SPHERE_GENERATOR_H
#include <thread>

#include "cuda_types/stream.h"
#include "graphics/gl_context_lock.h"
#include "sphere_generation/fwd.h"
namespace raw::sphere_generation {
class sphere_generator {
private:
	std::jthread worker_thread;
public:
	sphere_generator() = default;
	void generate(UI steps, cuda_types::cuda_stream& stream, icosahedron_data_manager& source, graphics::graphics_data& data);
	void sync();
};
} // namespace raw::sphere_generation

#endif // SPACE_EXPLORER_SPHERE_GENERATOR_H
