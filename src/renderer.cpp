//
// Created by progamers on 7/4/25.
//
#include "renderer.h"

#include "scene.h"
#define AM_POINT_LIGHTS 5
namespace raw {
renderer::renderer(const std::string &window_name)
	: window(window_name),
	  object_shader(make_shared<raw::shader>("shaders/objects/vertex_shader.glsl",
											 "shaders/objects/color_shader.frag")),
	  light_shader(make_shared<shader>("shaders/light/vertex_shader.glsl",
									   "shaders/light/color_shader.frag")),
	  outline_shader(make_shared<raw::shader>("shaders/outline/vertex_shader.glsl",
											  "shaders/outline/color_shader.frag")),
	  cube_object(object_shader),
	  light_cube(light_shader),
	  system(predef::generate_data_for_sim()),
	  sphere_mesh(make_shared<mesh>(predef::MAXIMUM_AMOUNT_OF_VERTICES,
									predef::MAXIMUM_AMOUNT_OF_INDICES)) {
	inst_renderer.set_data(sphere_mesh);
	system.setup_model(inst_renderer.get_instance_vbo());
	gen.generate(sphere_mesh->get_vbo(), sphere_mesh->get_ebo(), predef::BASIC_STEPS, 1);
	// that's still the ugliest part of my code by far
	object_shader->use();
	object_shader->set_float("obj_mat.shininess", 32.0f);
	object_shader->set_vec3("dir_light.direction", 0.0f, -1.0f, 0.0f);
	object_shader->set_vec3("dir_light.ambient", 0.1f, 0.1f, 0.1f);
	object_shader->set_vec3("dir_light.diffuse", 0.8f, 0.8f, 0.8f);
	object_shader->set_vec3("dir_light.specular", 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < AM_POINT_LIGHTS; ++i) {
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].position",
								*(light_pos.begin() + i));
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].ambient", 0.05f, 0.05f,
								0.05f);
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].diffuse", 0.8f, 0.8f,
								0.8f);
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].specular", 1.0f, 1.0f,
								1.0f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].constant", 1.0f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].linear", 0.09f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].quadratic", 0.032f);
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

	raw::gl::MOUSE_GRAB(window.get(), true);
	raw::gl::RELATIVE_MOUSE_MODE(window.get(), true);

	raw::gl::RULE(GL_MULTISAMPLE);
	raw::gl::CLEAR_COLOR(0.1f, 0.1f, 0.1f, 1.0f);

	raw::gl::RULE(GL_DEPTH_TEST);
	raw::gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	raw::gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 4);
	raw::gl::RULE(GL_MULTISAMPLE);

	auto resolution = window.get_window_size();
	raw::gl::VIEW(0, 0, resolution.x, resolution.y);
	_clock.restart();
}

raw::shared_ptr<raw::shader> renderer::get_basic_shader() const {
	return object_shader;
}
std::vector<raw::shared_ptr<raw::shader>> renderer::get_all_shaders() const {
	return {object_shader, light_shader, outline_shader};
}

void renderer::render() {
	system.update_sim();
	window.clear();
	object_shader->set_vec3("point_lights[4].position", system[0].get().position);
	cube_object.set_shader(object_shader);
	cube_object.move(glm::vec3(0.0f, -2.0f, 0.0f));
	cube_object.scale(glm::vec3(15.0f, 0.2f, 15.0f));
	cube_object.draw();

	cube_object.set_shader(object_shader);
	for (auto cube_position : cube_positions) {
		cube_object.move(cube_position);
		cube_object.draw();
	}

	light_shader->use();
	for (auto light_cube_pos : light_pos) {
		light_cube.move(light_cube_pos);
		light_cube.scale(glm::vec3(0.2f));
		light_cube.draw();
	}

	inst_renderer.draw(object_shader, system.amount());

	window.update();
}

} // namespace raw