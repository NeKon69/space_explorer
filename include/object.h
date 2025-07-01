//
// Created by progamers on 7/1/25.
//

#ifndef SPACE_EXPLORER_OBJECT_H
#define SPACE_EXPLORER_OBJECT_H
#include "helper_macros.h"
namespace raw {
class object {
private:
    // Idk is it really that smart to like "track" those buffers via pointers, but that's what you pay if you want to copy the object
	UI* vao;
    UI* vbo;
    UI* ebo;
    bool* deleted_buffers;
public:
	object();
	object(const float* vertices, const UI* indices);
	object(const object&);
	object(object&&) noexcept;
	object& operator=(const object&);
	object& operator=(object&&) noexcept;
	~object() noexcept;
	void draw() const;
	void setup_object(const float* vertices, const UI* indices);
};
} // namespace raw

#endif // SPACE_EXPLORER_OBJECT_H
