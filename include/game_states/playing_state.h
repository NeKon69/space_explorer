//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_PLAYING_STATE_H
#define SPACE_EXPLORER_PLAYING_STATE_H

#include <glm/glm.hpp>

#include "core/camera/camera.h"
#include "core/camera/movement_state.h"
#include "core/camera/player_controller.h"
#include "core/game_state.h"
#include "n_body/interaction_system.h"
#include "n_body/simulation_state.h"
#include "sphere_generation/mesh_generator.h"
#include "z_unused/objects/cube.h"
#include "graphics/mesh.h"

namespace raw::game_states {
	class playing_state : public raw::core::game_state {
	private:
		raw::shared_ptr<raw::shader> object_shader;
		raw::shared_ptr<raw::shader> light_shader;
		raw::core::camera::movement_state move_state;

		static constexpr std::initializer_list<glm::vec3> light_pos = {
		glm::vec3(2.5, 2.5, 5), glm::vec3(-5, -5, 10), glm::vec3(0, -5, -5), glm::vec3(-5, 5, 5)};

	raw::cube light_cube;

		raw::shared_ptr<graphics::mesh> sphere_mesh;
		raw::sphere_generation::icosahedron_generator gen;
	bool						   dir_light = false;
	raw::simulation_state		   sim_state;
	raw::n_body::interaction_system<float> system;

		bool pressed_o = false;

	[[nodiscard]] raw::shared_ptr<raw::shader>				get_basic_shader() const;
	[[nodiscard]] std::vector<raw::shared_ptr<raw::shader>> get_all_shaders() const;

		raw::core::camera::camera camera;

		raw::core::camera::player_controller controller;
		raw::core::clock clock;

		bool is_active = true;

	void init() const;

public:
	explicit playing_state(glm::uvec2 window_size = {2560, 1440});
	~playing_state() override;
	bool								handle_input() override;
	void update(const raw::core::time &delta_time) override;

		void								draw(raw::rendering::renderer& renderer) override;
	bool								active() override;
	[[nodiscard]] raw::rendering::queue build_rendering_queue() const;
};

} // namespace raw::game_states
#endif // SPACE_EXPLORER_PLAYING_STATE_H
