//
// Created by progamers on 7/4/25.
//

#pragma once

#include <glm/glm.hpp>

#include "core/camera/camera.h"
#include "core/camera/movement_state.h"
#include "core/camera/player_controller.h"
#include "core/game_state.h"
#include "graphics/mesh.h"
#include "n_body/n_body_factory.h"
#include "sphere_generation/cuda/sphere_generator.h"
#include "sphere_generation/cuda/sphere_resource_manager.h"
#include "z_unused/objects/cube.h"

namespace raw::game_states {
class playing_state : public raw::core::game_state {
private:
	std::shared_ptr<raw::rendering::shader::shader> object_shader;
	std::shared_ptr<raw::rendering::shader::shader> light_shader;
	raw::core::camera::movement_state				move_state;

	static constexpr std::initializer_list<glm::vec3> light_pos = {
		glm::vec3(2.5, 2.5, 5), glm::vec3(-5, -5, 10), glm::vec3(0, -5, -5), glm::vec3(-5, 5, 5)};

	raw::z_unused::objects::cube light_cube;

	// For now let's just store the stream locally
	std::shared_ptr<device_types::i_queue>						  stream;
	std::shared_ptr<graphics::mesh>								  sphere_mesh;
	std::shared_ptr<sphere_generation::i_sphere_resource_manager> sphere_manager;
	std::shared_ptr<sphere_generation::i_sphere_generator>		  sphere_gen;
	std::shared_ptr<entity_management::entity_manager>			  entity_manager;
	bool														  dir_light = false;
	graphics::instanced_data_buffer								  render_buffer;
	std::unique_ptr<raw::n_body::interaction_system<float> >	  interaction_system;

	bool pressed_o = false;

	[[nodiscard]] std::shared_ptr<raw::rendering::shader::shader> get_basic_shader() const;

	[[nodiscard]] std::vector<std::shared_ptr<raw::rendering::shader::shader> > get_all_shaders()
		const;

	raw::core::camera::camera camera;

	raw::core::camera::player_controller controller;
	raw::core::clock					 clock;

	bool is_active = true;

	void init() const;

public:
	explicit playing_state(graphics::graphics_data& graphics_data,
						   glm::uvec2				window_size = {2560, 1440});
	~playing_state() override;
	bool handle_input() override;
	void update(const raw::core::time& delta_time) override;

	void								draw(raw::rendering::renderer& renderer) override;
	bool								active() override;
	[[nodiscard]] raw::rendering::queue build_rendering_queue() const;
};

} // namespace raw::game_states
