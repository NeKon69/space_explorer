//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_GENERATION_CONTEST_H
#define SPACE_EXPLORER_GENERATION_CONTEST_H
#include "sphere_generation/cuda/fwd.h"
#include "sphere_generation/fwd.h"

namespace raw::sphere_generation {
class generation_context {
private:
	i_sphere_resource_manager* manager;
	friend class i_sphere_resource_manager;
	friend class cuda::sphere_resource_manager;

protected:
	generation_context(i_sphere_resource_manager* mgr, uint32_t vbo, uint32_t ebo);

public:
	~generation_context();
	generation_context(const generation_context& other)				   = delete;
	generation_context(generation_context&& other) noexcept			   = default;
	generation_context& operator=(const generation_context& other)	   = delete;
	generation_context& operator=(generation_context&& other) noexcept = default;
};
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_GENERATION_CONTEST_H
