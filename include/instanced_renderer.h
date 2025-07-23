//
// Created by progamers on 7/23/25.
//

#ifndef SPACE_EXPLORER_INSTANCED_RENDERER_H
#define SPACE_EXPLORER_INSTANCED_RENDERER_H
#include "instance_data.h"
#include "mesh.h"
namespace raw {
template<typename T>
class instanced_renderer {
private:
	shared_ptr<mesh>				   _mesh;
	unique_ptr<UI, deleter::gl_buffer> instance_vbo;
	bool							   generated = false;

	void generate() {
		if (generated)
			return;
		glGenBuffers(1, *instance_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *instance_vbo);
		_mesh->bind();
		T::setup_instance_arttr(_mesh->attr_num());
		_mesh->unbind();
		generated = true;
	}

public:
	instanced_renderer() = default;
	explicit instanced_renderer(const shared_ptr<mesh>& mesh) : _mesh(mesh) {
		generate();
	}
	void set_data(const shared_ptr<mesh>& mesh) {
		_mesh = mesh;
		generate();
	}
	void draw(UI amount) const {
		_mesh->bind();
		glDrawElementsInstanced(GL_TRIANGLES, _mesh->get_index_count(), GL_UNSIGNED_INT, nullptr,
								static_cast<int>(amount));
		_mesh->unbind();
	}
};
} // namespace raw
#endif // SPACE_EXPLORER_INSTANCED_RENDERER_H
