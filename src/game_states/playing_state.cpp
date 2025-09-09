//
// Created by progamers on 7/4/25.
//
#include "game_states/playing_state.h"

#include "rendering/render_command.h"
#include "rendering/renderer.h"
#include "window/gl_window.h"

namespace raw::game_states {
namespace predef {
static constexpr auto AM_POINT_LIGHTS = 5;
}

void playing_state::init() const {
	// Setup all those wierd ass shaders...
	// that's still the ugliest part of my code by far
	object_shader->use();
	object_shader->set_float("obj_mat.shininess", 32.0f);
	object_shader->set_vec3("dir_light.direction", 0.0f, -1.0f, 0.0f);
	object_shader->set_vec3("dir_light.ambient", 0.1f, 0.1f, 0.1f);
	object_shader->set_vec3("dir_light.diffuse", 0.8f, 0.8f, 0.8f);
	object_shader->set_vec3("dir_light.specular", 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < predef::AM_POINT_LIGHTS; ++i) {
		object_shader->set_vec3(std::format("point_lights[{}].position", i),
								*(light_pos.begin() + i));
		object_shader->set_vec3(std::format("point_lights[{}].ambient", i), 0.05f, 0.05f, 0.05f);
		object_shader->set_vec3(std::format("point_lights[{}].diffuse", i), 0.8f, 0.8f, 0.8f);
		object_shader->set_vec3(std::format("point_lights[{}].specular", i), 1.0f, 1.0f, 1.0f);
		object_shader->set_float(std::format("point_lights[{}].constant", i), 1.0f);
		object_shader->set_float(std::format("point_lights[{}].linear", i), 0.09f);
		object_shader->set_float(std::format("point_lights[{}].quadratic", i), 0.032f);
	}
	object_shader->set_vec3("sp_light.ambient", 0.0f, 0.0f, 0.0f);
	object_shader->set_vec3("sp_light.diffuse", 1.0f, 1.0f, 1.0f);
	object_shader->set_vec3("sp_light.specular", 1.0f, 1.0f, 1.0f);
	object_shader->set_float("sp_light.cut_off", glm::cos(glm::radians(12.5f)));
	object_shader->set_float("sp_light.outer_cut_off", glm::cos(glm::radians(17.5f)));
	object_shader->set_float("sp_light.constant", 1.0f);
	object_shader->set_float("sp_light.linear", 0.09f);
	object_shader->set_float("sp_light.quadratic", 0.032f);

	object_shader->set_vec3("obj_mat.ambient", 0.1f, 0.1f, 0.1f);
	object_shader->set_vec3("obj_mat.diffuse", 1.0f, 0.5f, 0.31f);
	object_shader->set_vec3("obj_mat.specular", 0.5f, 0.5f, 0.5f);

	light_shader->use();
	light_shader->set_vec3("lightColor", 1, 1, 1);

	raw::window::gl::RULE(GL_MULTISAMPLE);
	raw::window::gl::CLEAR_COLOR(0.1f, 0.1f, 0.1f, 1.0f);

	raw::window::gl::RULE(GL_DEPTH_TEST);
	raw::window::gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	raw::window::gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 4);
	raw::window::gl::RULE(GL_MULTISAMPLE);
}

playing_state::playing_state(graphics::graphics_data& graphics_data, glm::uvec2 window_size)
	: object_shader(std::make_shared<raw::rendering::shader::shader>(
		  "shaders/objects/vertex_shader.glsl", "shaders/objects/color_shader.frag")),
	  light_shader(std::make_shared<raw::rendering::shader::shader>(
		  "shaders/light/vertex_shader.glsl", "shaders/light/color_shader.frag")),
	  light_cube(object_shader),

	  stream(std::make_shared<device_types::cuda::cuda_stream>()),
	  sphere_mesh(std::make_shared<raw::graphics::mesh>(
		  sphere_generation::predef::MAXIMUM_AMOUNT_OF_VERTICES,
		  sphere_generation::predef::MAXIMUM_AMOUNT_OF_INDICES)),
	  sphere_manager(std::make_shared<sphere_generation::cuda::sphere_resource_manager>(
		  sphere_mesh->get_vbo(), sphere_mesh->get_ebo(), stream)),
	  sphere_gen(std::make_shared<sphere_generation::cuda::sphere_generator>()),
	  render_buffer(sphere_mesh->get_vao(), sphere_mesh->attr_num(), 1000),
	  interaction_system(n_body::n_body_factory<float, backend::CUDA>::create(
		  render_buffer, stream, 1000, n_body::predef::generate_data_for_sim(), graphics_data)),
	  camera(),
	  controller(camera) {
	sphere_mesh->unbind();
	sphere_gen->generate(7, stream, sphere_manager, graphics_data);
	camera.set_window_resolution(window_size.x, window_size.y);
	init();
}

raw::rendering::queue playing_state::build_rendering_queue() const {
	raw::rendering::queue queue;
	{
		raw::rendering::command cmd;
		cmd.shader	  = object_shader;
		cmd.mesh	  = sphere_mesh;
		cmd.inst_data = raw::rendering::instance_data {
			render_buffer.get_vbo(), static_cast<int>(interaction_system->get_amount())};
		queue.push_back(cmd);
	}
	return queue;
}

void playing_state::update(const raw::core::time& delta_time) {
	interaction_system->update_simulation();
	controller.update(move_state, delta_time.val);
}

bool playing_state::handle_input() {
	// Yew
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		if (event.type == SDL_EVENT_KEY_DOWN) {
			const auto& scancode = event.key.scancode;
			// I have no idea tf am I doing with my life, but what i do know, that this should be
			// replaced with some settings instead of hardcoding each scancode
			// TODO: add some struct to store those things nicely so you can change what you press
			// Maybe i suppose it would be nice to use my `button` class
			// to do movement for example
			switch (scancode) {
			case SDL_SCANCODE_W:
				move_state.forward = true;
				break;
			case SDL_SCANCODE_A:
				move_state.left = true;
				break;
			case SDL_SCANCODE_S:
				move_state.backward = true;
				break;
			case SDL_SCANCODE_D:
				move_state.right = true;
				break;
			case SDL_SCANCODE_ESCAPE:
				return true;
			case SDL_SCANCODE_TAB:
				move_state.down = true;
				break;
			case SDL_SCANCODE_SPACE:
				move_state.up = true;
				break;
			case SDL_SCANCODE_O:
				if (pressed_o)
					break;
				if (interaction_system->running()) {
					interaction_system->pause();
				} else {
					interaction_system->resume();
				}
				break;
			case SDL_SCANCODE_T:
				dir_light = !dir_light;
				break;
			default:
				break;
			}
		} else if (event.type == SDL_EVENT_KEY_UP) {
			const auto& scancode = event.key.scancode;
			switch (scancode) {
			case SDL_SCANCODE_W:
				move_state.forward = false;
				break;
			case SDL_SCANCODE_A:
				move_state.left = false;
				break;
			case SDL_SCANCODE_S:
				move_state.backward = false;
				break;
			case SDL_SCANCODE_D:
				move_state.right = false;
				break;
			case SDL_SCANCODE_TAB:
				move_state.down = false;
				break;
			case SDL_SCANCODE_SPACE:
				move_state.up = false;
				break;
			case SDL_SCANCODE_O:
				pressed_o = false;
				break;
			default:
				break;
			}
		} else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
			camera.adjust_fov(event.wheel.y);
		} else if (event.type == SDL_EVENT_MOUSE_MOTION) {
			auto x = event.motion.xrel * core::camera::predef::SENSITIVITY;
			auto y = event.motion.yrel * core::camera::predef::SENSITIVITY;
			camera.set_rotation(x, y);
		}
	}
	return false;
}

void playing_state::draw(raw::rendering::renderer& renderer) {
	auto queue = build_rendering_queue();
	sphere_gen->sync();
	renderer.render(queue, camera);
}

bool playing_state::active() {
	return is_active;
}

playing_state::~playing_state() {
	std::cout << "[Debug] Quiting the playing state\n";
}

} // namespace raw::game_states