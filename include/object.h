//
// Created by progamers on 7/1/25.
//

#ifndef SPACE_EXPLORER_OBJECT_H
#define SPACE_EXPLORER_OBJECT_H

// I really wonder right now, what should I better do? wrap all opengl calls like  glDeleteBuffers,
// glBindBuffer, etc... in some class, and check for the errors, or leave it as it is, and hope
// everything works, IDK.

#include <raw_memory.h>

#include "helper_macros.h"
#include "shader.h"
namespace raw {
struct gl_data_deleter_buffer {
	explicit gl_data_deleter_buffer(UI const* data) {
		glDeleteBuffers(1, data);
		delete data;
	}
};

struct gl_data_deleter_array {
	explicit gl_data_deleter_array(UI const* data) {
		glDeleteBuffers(1, data);
		delete data;
	}
};

class object {
private:
	// my own kiddie, it's ugly but i SOOO like it)))
	// look how clean it looks!!!
	raw::unique_ptr<UI, gl_data_deleter_array>	vao;
	raw::unique_ptr<UI, gl_data_deleter_buffer> vbo;
	raw::unique_ptr<UI, gl_data_deleter_buffer> ebo;
	raw::unique_ptr<int>						indices_size;

protected:
	glm::mat4					 transformation = glm::mat4(1.0f);
	raw::shared_ptr<raw::shader> shader;

public:
	object(object&&) noexcept;
	object& operator=(object&&) noexcept;

	void rotate(float degree, const glm::vec3& rotation);
	void move(const glm::vec3& vec);
	void scale(const glm::vec3& factor);
	void reset();
	void set_shader(const raw::shared_ptr<raw::shader>& sh);
	/**
	 * \brief
	 * draw the object
	 * \param reset should matrix reset? defaults to true
	 */
	void draw(bool reset = true);

	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	object(const T vertices, const Y indices)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)), indices_size(new int(0)) {
		setup_object(vertices, indices);
	}

	/**
	 * \brief
	 * that function can only be called via std/classes that have `begin` method (for example, you
	 * can use it with vector, but can't with c-style arrays)
	 */
	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	void setup_object(const T vertices, const Y indices) {
		// still require some manual setup
		*indices_size = indices.size();
		glGenVertexArrays(1, vao.get());
		glGenBuffers(1, vbo.get());
		glGenBuffers(1, ebo.get());

		glBindVertexArray(*vao);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), std::begin(vertices),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(),
					 std::begin(indices), GL_STATIC_DRAW);

		// positions and normals
		constexpr GLsizei stride = 6 * sizeof(float);
		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
		glEnableVertexAttribArray(0);
		// normal attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindVertexArray(0);
	}

protected:
	void __draw() const;
};
} // namespace raw

#endif // SPACE_EXPLORER_OBJECT_H
