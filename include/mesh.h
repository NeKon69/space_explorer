//
// Created by progamers on 6/22/25.
//

#ifndef SPACE_EXPLORER_MESH_H
#define SPACE_EXPLORER_MESH_H

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace raw {

using UI = unsigned int;
template<typename T>
using vec = std::vector<T>;


// fwd declaration of shader class for use in mash class
class shader;

struct vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 tex_coords;
};

struct texture {
	unsigned int id;
	std::string	 type;
};

class mesh {
public:
    // no default constructors, cause it just won't make any sense
    mesh() = delete;

    mesh(vec<vertex>& vcs, vec<UI>& ids, vec<texture>& txs);
    // no copying, cause well, mf waste of space would be crazy bad
    mesh            (mesh&) = delete;
    mesh operator=  (mesh&) = delete;

    void draw(shader& shader);
    vec<vertex>     get_vertices()  { return vertices;  }
    vec<UI>         get_indices()   { return indices;   }
    vec<texture>    get_textures()  { return textures;  }

private:
    vec<vertex>     vertices;
    vec<UI>         indices;
    vec<texture>    textures;
    unsigned int    vao = 0, vbo = 0, ebo = 0;

	void	        setup_mash();

};
} // namespace raw

#endif // SPACE_EXPLORER_MESH_H
