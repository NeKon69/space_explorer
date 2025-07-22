//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_RENDERER_H
#define SPACE_EXPLORER_RENDERER_H

#include <raw_memory.h>

#include "clock.h"
#include "gl_window.h"
#include "n_body/drawable_space_object.h"
#include "n_body/interaction_system.h"
#include "objects/cube.h"
#include "objects/sphere.h"
#include "shader.h"
namespace raw {
// I am not entirely sure what to put here, so I guess I just put something and then expend if
// needed
class renderer {
private:
	raw::gl_window window;

	raw::shared_ptr<raw::shader>					  object_shader;
	raw::shared_ptr<raw::shader>					  light_shader;
	raw::shared_ptr<raw::shader>					  outline_shader;
	raw::clock										  _clock;
	static constexpr std::initializer_list<glm::vec3> light_pos = {
		glm::vec3(2.5, 2.5, 5), glm::vec3(-5, -5, 10), glm::vec3(0, -5, -5), glm::vec3(-5, 5, 5)};
	static constexpr std::initializer_list<glm::vec3> cube_positions = {
		glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 5.0f, -15.0f)};

	static constexpr std::initializer_list<glm::vec3> sphere_positions = {{2.f, 0.5, 0.5}};

	raw::cube				   cube_object;
	raw::cube				   light_cube;
	raw::sphere				   sphere;
	raw::drawable_space_object<double> sphere_obj;
	raw::interaction_system<double>	   system;
	bool					   dir_light = true;
	friend class event_handler;
	friend class scene;

	[[nodiscard]] raw::shared_ptr<raw::shader>				get_basic_shader() const;
	[[nodiscard]] std::vector<raw::shared_ptr<raw::shader>> get_all_shaders() const;

public:
	explicit renderer(const std::string &window_name = "Mike Hawk");
	[[nodiscard]] bool is_window_running() const {
		return window.is_running();
	}
	void set_window_running(bool state) {
		window.set_running(state);
	}
	[[nodiscard]] bool get_dir_light() const noexcept {
		return dir_light;
	}
	void set_dir_light(bool state) noexcept {
		dir_light = state;
	}
	void render();
};

} // namespace raw

#endif // SPACE_EXPLORER_RENDERER_H
