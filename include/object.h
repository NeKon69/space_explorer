//
// Created by progamers on 7/1/25.
//

#ifndef SPACE_EXPLORER_OBJECT_H
#define SPACE_EXPLORER_OBJECT_H
#include <raw_memory.h>

#include "helper_macros.h"
#include "shader.h"
namespace raw {
class object {
private:
	/*
	 * \brief
	 *  frick std man, I'll use my own smart pointers, and no one can say shit to me, cause
	 *  see benchmark and my tests on that project, I fucking demolish std
	 */
	raw::shared_ptr<UI> vao;
	raw::shared_ptr<UI> vbo;
	raw::shared_ptr<UI> ebo;
    raw::shared_ptr<UI>	indices_size;
public:
    object() = delete;
	object(const object&) = default;
	object(object&&) noexcept;
	object& operator=(const object&);
	object& operator=(object&&) noexcept;
	~object() noexcept;

    template<typename T, typename Y>
    object(const T vertices, const Y indices) : vao(raw::make_shared<UI>(0)),
                                                vbo(raw::make_shared<UI>(0)),
                                                ebo(raw::make_shared<UI>(0)),
                                                indices_size(raw::make_shared<UI>(0)) {
        setup_object(vertices, indices);
    }

    /*
    * \brief
    * that function can only be called via std/classes that have `begin` method (for example, you can use it with vector, but can't with c-style arrays)
    */
    template<typename T, typename Y>
	void setup_object(const T vertices, const Y indices) {
        *indices_size = indices.size();
        glGenVertexArrays(1, vao.get());
        glGenBuffers(1, vbo.get());
        glGenBuffers(1, ebo.get());

        glBindVertexArray(*vao);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), std::begin(vertices), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), std::begin(indices),
                     GL_STATIC_DRAW);

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
