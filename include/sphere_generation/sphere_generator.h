//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_SPHERE_GENERATOR_H
#define SPACE_EXPLORER_SPHERE_GENERATOR_H
#include <thread>

#include "cuda_types/stream.h"
#include "graphics/gl_context_lock.h"
#include "sphere_generation/fwd.h"
/**
 * Start asynchronous sphere generation with the given subdivision steps.
 *
 * Launches background work on the internal worker thread to generate or update
 * sphere geometry using the provided CUDA stream and icosahedron/graphics resources.
 * The operation runs asynchronously; call sync() to wait for completion.
 * @param steps Number of subdivision steps to apply to the base icosahedron.
 */

/**
 * Block until the internal worker thread completes any outstanding work.
 *
 * If no background work is running, returns immediately.
 */
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
