//
// Created by progamers on 6/22/25.
//

#ifndef SPACE_EXPLORER_MESH_H
#define SPACE_EXPLORER_MESH_H

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "helper_macros.h"

namespace raw {

// fwd declaration of shader class for use in mash class
class shader;

struct vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 tex_coords;
};

struct texture {
	unsigned int id;
	std::string	 type;
	std::string	 path;
};

class mesh {
public:
	mesh() = default;
	mesh(const vec<vertex>& vcs, const vec<UI>& ids, const vec<texture>& txs);

	void		draw(shader& shader) const;
	vec<vertex> get_vertices() const {
		return vertices;
	}
	vec<UI> get_indices() const {
		return indices;
	}
	vec<texture> get_textures() const {
		return textures;
	}

private:
	vec<vertex>	 vertices;
	vec<UI>		 indices;
	vec<texture> textures;
	unsigned int vao = 0, vbo = 0, ebo = 0;

	void setup_mash();
};
} // namespace raw

#endif // SPACE_EXPLORER_MESH_H
