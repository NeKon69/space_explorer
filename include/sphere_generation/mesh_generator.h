//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include <raw_memory.h>

#include <array>
#include <glm/glm.hpp>

#include "cuda_from_gl_data.h"
#include "helper_macros.h"
namespace raw {

namespace predef {
PASSIVE_VALUE BASIC_RADIUS				= 1.0f;
PASSIVE_VALUE BASIC_STEPS				= 5U;
PASSIVE_VALUE MAX_STEPS					= 10U;
PASSIVE_VALUE BASIC_AMOUNT_OF_TRIANGLES = 20U;
} // namespace predef

// this class serves as a nice thing to warp around hard things in generating sphere from
// icosahedron
class icosahedron_generator {
private:
	raw::unique_ptr<cuda_from_gl_data> vertices_handle;
	raw::unique_ptr<cuda_from_gl_data> indices_handle;
	void							   assign_data(UI* indices, glm::vec3* vertices, float radius);

public:
	icosahedron_generator() = default;
	icosahedron_generator(UI vbo, UI ebo, UI steps = predef::BASIC_STEPS,
						  float radius = predef::BASIC_RADIUS);
	void									   generate(UI vbo, UI ebo, UI steps, float radius);
	static constexpr std::array<glm::vec3, 12> generate_icosahedron_vertices(float radius);
	static constexpr std::array<UI, 60>		   generate_icosahedron_indices();
	static constexpr std::pair<std::array<glm::vec3, 12>, std::array<UI, 60>>
	generate_icosahedron_data(float radius);
};

} // namespace raw
#endif // SPACE_EXPLORER_MESH_GENERATOR_H
