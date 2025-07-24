//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_SCENE_H
#define SPACE_EXPLORER_SCENE_H

#include <glm/glm.hpp>

#include "camera.h"
#include "event_handler.h"
#include "input_manager.h"
#include "n_body/interaction_system.h"
#include "objects/cube.h"
#include "rendering/render_command.h"
#include "rendering/renderer.h"
#include "sphere_generation/mesh_generator.h"
namespace raw {
class scene {
private:
	raw::rendering::renderer renderer;

	raw::input_manager manager;

	raw::shared_ptr<raw::shader>					  object_shader;
	raw::shared_ptr<raw::shader>					  light_shader;
	raw::shared_ptr<raw::shader>					  outline_shader;
	static constexpr std::initializer_list<glm::vec3> light_pos = {
		glm::vec3(2.5, 2.5, 5), glm::vec3(-5, -5, 10), glm::vec3(0, -5, -5), glm::vec3(-5, 5, 5)};

	raw::cube					   light_cube;
	raw::interaction_system<float> system;
	raw::shared_ptr<mesh>		   sphere_mesh;
	raw::icosahedron_generator	   gen;
	bool						   dir_light = true;
	friend class event_handler;
	friend class scene;

	[[nodiscard]] raw::shared_ptr<raw::shader>				get_basic_shader() const;
	[[nodiscard]] std::vector<raw::shared_ptr<raw::shader>> get_all_shaders() const;

	raw::camera camera;
	friend class event_handler;
	friend class renderer;

public:
	explicit scene(const std::string& window_name = "Mike Hawk");
	raw::rendering::queue build_rendering_queue() const;
	void				  run();
};

} // namespace raw
#endif // SPACE_EXPLORER_SCENE_H
