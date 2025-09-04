//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_GENERATION_CONTEST_H
#define SPACE_EXPLORER_GENERATION_CONTEST_H
#include "sphere_generation/fwd.h"

namespace raw::sphere_generation {
class generation_context {
private:
	icosahedron_data_manager& manager;
	friend class icosahedron_data_manager;

protected:
	generation_context(icosahedron_data_manager& mgr, UI vbo, UI ebo);

public:
	~generation_context();
	generation_context(const generation_context& other)				   = delete;
	generation_context(generation_context&& other) noexcept			   = default;
	generation_context& operator=(const generation_context& other)	   = delete;
	generation_context& operator=(generation_context&& other) noexcept = default;
};
} // namespace raw::sphere_generation
#endif // SPACE_EXPLORER_GENERATION_CONTEST_H
