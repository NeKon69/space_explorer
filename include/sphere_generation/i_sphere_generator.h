//
// Created by progamers on 9/5/25.
//

#ifndef SPACE_EXPLORER_I_SPHERE_GENERATOR_H
#define SPACE_EXPLORER_I_SPHERE_GENERATOR_H

#include <memory>
#include <thread>

#include "cuda_types/fwd.h"
#include "graphics/gl_context_lock.h"
#include "sphere_generation/fwd.h"

namespace raw::sphere_generation {
class i_sphere_generator {
protected:
	std::jthread worker_thread;

public:
	virtual void generate(uint32_t steps, cuda_types::cuda_stream& stream,
						  std::shared_ptr<i_sphere_resource_manager> source,
						  graphics::graphics_data&					 data) = 0;
	void		 sync();
	virtual ~i_sphere_generator() = default;
	i_sphere_generator()		  = default;
};

} // namespace raw::sphere_generation

#endif // SPACE_EXPLORER_I_SPHERE_GENERATOR_H
