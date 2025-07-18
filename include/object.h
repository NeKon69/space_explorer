//
// Created by progamers on 7/1/25.
//

#ifndef SPACE_EXPLORER_OBJECT_H
#define SPACE_EXPLORER_OBJECT_H

// I really wonder right now, what should I better do? wrap all opengl calls like  glDeleteBuffers,
// glBindBuffer, etc... in some class, and check for the errors, or leave it as it is, and hope
// everything works, IDK.

#include <glad/glad.h>
#include <raw_memory.h>
#include <glm/vec3.hpp>

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

namespace drawing_method {
void				  basic(UI* vao, UI indices_size);
void				  lines(UI* vao, UI indices_size);
void				  points(UI* vao, UI indices_size);
void				  transparent_alpha(UI* vao, UI indices_size);
void				  always_visible(UI* vao, UI indices_size);
void				  backface_cull(UI* vao, UI indices_size);
void				  polygon_offset_fill(UI* vao, UI indices_size);
void				  stencil_mask_equal_1(UI* vao, UI indices_size);
void				  depth_write_disabled(UI* vao, UI indices_size);
void				  color_mask_red_only(UI* vao, UI indices_size);
void				  blend_additive(UI* vao, UI indices_size);
inline PASSIVE_VALUE& drawing_method = basic;
} // namespace drawing_method

class object {
private:
	// my own kiddie, it's ugly but i SOOO like it)))
	// look how clean it looks!!!
	raw::unique_ptr<UI, gl_data_deleter_array>	vao;
	raw::unique_ptr<UI, gl_data_deleter_buffer> vbo;
	raw::unique_ptr<UI, gl_data_deleter_buffer> ebo;
	size_t										indices_size;

	void gen_opengl_data();

protected:
	glm::mat4					 transformation = glm::mat4(1.0f);
	raw::shared_ptr<raw::shader> shader;

public:
	object(object&&) noexcept;
	object& operator=(object&&) noexcept;

	void	  rotate(float degree, const glm::vec3& axis);
	void	  move(const glm::vec3& vec);
	void	  scale(const glm::vec3& factor);
	glm::mat4 get_mat() const {
		return transformation;
	}
	void			 reset();
	[[nodiscard]] UI get_vbo() const {
		return *vbo.get();
	}
	void set_shader(const raw::shared_ptr<raw::shader>& sh);
	/**
	 * \brief
	 * draw the object
	 * \param reset should matrix reset? defaults to true
	 */
	void draw(decltype(drawing_method::drawing_method) drawing_method = drawing_method::basic,
			  bool									   reset		  = true);

	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	object(const T& vertices, const Y& indices)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)), indices_size(0) {
		setup_object(vertices, indices);
	}
	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	object(const T& vertices, const size_t amount_of_vertices, const Y& indices,
		   const size_t amount_of_indices)
		: vao(new UI(0)), vbo(new UI(0)), ebo(new UI(0)), indices_size(amount_of_indices) {
		setup_object(vertices, amount_of_vertices, indices, amount_of_indices);
	}
	object() = default;

	/**
	 * \brief
	 * that function can only be called via std/classes that have `begin` method (for example, you
	 * can use it with vector, but can't with c-style arrays)
	 */

	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>
	void setup_object(const T& vertices, const Y& indices) {
		// still require some manual setup
		indices_size = indices.size();
		gen_opengl_data();

		glBindVertexArray(*vao);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), std::data(vertices),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(),
					 std::data(indices), GL_STATIC_DRAW);

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

	template<typename T, typename Y>
		requires std::ranges::range<T> && std::ranges::range<Y>

	void setup_object(const T& vertices, const size_t amount_of_vertices, const Y& indices,
					  const size_t amount_of_indices) {
		gen_opengl_data();

		glBindVertexArray(*vao);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * amount_of_vertices, std::data(vertices),
					 GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * amount_of_indices,
					 std::data(indices), GL_STATIC_DRAW);

		// here we accept only the position in the space
		constexpr GLsizei stride = 3 * sizeof(float);
		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
		glEnableVertexAttribArray(0);

		glBindVertexArray(0);
	}

protected:
	void __draw() const;
};
} // namespace raw

#endif // SPACE_EXPLORER_OBJECT_H
