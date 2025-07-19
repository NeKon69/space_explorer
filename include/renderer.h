//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_RENDERER_H
#define SPACE_EXPLORER_RENDERER_H

#include <raw_memory.h>

#include "gl_window.h"
#include "objects/cube.h"
#include "shader.h"
#include "objects/sphere.h"
namespace raw {
// I am not entirely sure what to put here, so I guess I just put something and then expend if
// needed
class renderer {
private:
	raw::gl_window window;

	raw::shared_ptr<raw::shader>					  object_shader;
	raw::shared_ptr<raw::shader>					  light_shader;
	raw::shared_ptr<raw::shader>					  outline_shader;
	static constexpr std::initializer_list<glm::vec3> light_pos = {
		glm::vec3(2.5, 2.5, 5), glm::vec3(-5, -5, 10), glm::vec3(0, -5, -5), glm::vec3(-5, 5, 5)};
	static constexpr std::initializer_list<glm::vec3> cube_positions = {
		glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 5.0f, -15.0f)};
	static constexpr std::initializer_list<glm::vec3> sphere_positions = {{2.f, 0.5, 0.5}};

	raw::cube	 cube_object;
	raw::cube	 light_cube;
	raw::sphere sphere;
    bool dir_light = true;
	friend class event_handler;
	friend class scene;

public:
	renderer(std::string window_name = "Mike Hawk");
	void render();
};

} // namespace raw

#endif // SPACE_EXPLORER_RENDERER_H
