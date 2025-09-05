//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_SPHERE_GENERATOR_H
#define SPACE_EXPLORER_SPHERE_GENERATOR_H
#include <thread>

#include "cuda_types/stream.h"
#include "fwd.h"
#include "graphics/gl_context_lock.h"
#include "sphere_generation/i_sphere_generator.h"
namespace raw::sphere_generation::cuda {
class sphere_generator : public i_sphere_generator {
public:
	sphere_generator() = default;
	void generate(uint32_t steps, cuda_types::cuda_stream& stream,
				  std::shared_ptr<i_sphere_resource_manager> source,
				  graphics::graphics_data&					 data) override;
};
} // namespace raw::sphere_generation::cuda

#endif // SPACE_EXPLORER_SPHERE_GENERATOR_H
