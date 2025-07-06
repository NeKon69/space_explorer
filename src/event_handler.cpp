//
// Created by progamers on 7/4/25.
//
#include "event_handler.h"

#include <iostream>

#include "camera.h"
#include "scene.h"
namespace raw {

event_handler::event_handler() {}

void event_handler::_setup_keys(raw::scene *scene) {
	// expand this when you add shaders/uniforms
	auto update_all_view_uniforms = [scene]() {
		update_uniform_for_shaders("view", scene->camera.view(), scene->renderer.object_shader,
								   scene->renderer.light_shader, scene->renderer.outline_shader);
		update_uniform(scene->renderer.object_shader, "sp_light.position", scene->camera.pos());
		update_uniform(scene->renderer.object_shader, "sp_light.direction", scene->camera.front());
	};
	// I am in hell
	auto create_camera_move_callback = [this, scene, update_all_view_uniforms](
										   decltype(raw::camera_move::MOVE_FUNCTION) direction) {
		return [this, direction, scene, update_all_view_uniforms]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / predef::UPDATE_MOVE_TIME))) {
				scene->camera.move(direction, update_all_view_uniforms);
				clock_callback.restart();
			}
		};
	};

	buttons[SDL_SCANCODE_TAB] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::DOWN));
	buttons[SDL_SCANCODE_SPACE] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::UP));
	buttons[SDL_SCANCODE_LEFT] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::LEFT));
	buttons[SDL_SCANCODE_RIGHT] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::RIGHT));
	buttons[SDL_SCANCODE_UP] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::FORWARD));
	buttons[SDL_SCANCODE_DOWN] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::BACKWARD));
	buttons[SDL_SCANCODE_W] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::FORWARD));
	buttons[SDL_SCANCODE_A] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::LEFT));
	buttons[SDL_SCANCODE_S] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::BACKWARD));
	buttons[SDL_SCANCODE_D] =
		raw::button(raw::func_type::HELD, create_camera_move_callback(raw::camera_move::RIGHT));

	auto T_CALLBACK = [scene]() {
		scene->renderer.object_shader->use();
		scene->renderer.dir_light = !scene->renderer.dir_light;
		update_uniform(scene->renderer.object_shader, "need_dir_light", scene->renderer.dir_light);
	};

	auto ESCAPE_CALLBACK = [scene]() {
		std::cout << "Don't close me mf!" << std::endl;
		scene->renderer.window.running = false;
	};

	buttons[SDL_SCANCODE_T]		 = raw::button(raw::func_type::PRESSED, T_CALLBACK);
	buttons[SDL_SCANCODE_ESCAPE] = raw::button(raw::func_type::PRESSED, ESCAPE_CALLBACK);
}

void event_handler::setup(raw::scene *scene) {
	auto update_front_view_uniforms = [scene]() {
		update_uniform_for_shaders("view", scene->camera.view(), scene->renderer.object_shader,
								   scene->renderer.light_shader, scene->renderer.outline_shader);
		update_uniform(scene->renderer.object_shader, "sp_light.direction", scene->camera.front());
	};
	auto update_projection_uniforms = [scene]() {
		update_uniform_for_shaders("projection", scene->camera.projection(),
								   scene->renderer.object_shader, scene->renderer.light_shader,
								   scene->renderer.outline_shader);
	};
	_setup_keys(scene);
	// pretty easy, right?
	events[SDL_EVENT_KEY_DOWN] = [this](SDL_Event event) {
		if (buttons.contains(event.key.scancode)) {
			buttons[event.key.scancode].press();
		}
	};
	events[SDL_EVENT_QUIT] = [this](SDL_Event) {
		if (buttons.contains(SDL_SCANCODE_ESCAPE)) {
			buttons[SDL_SCANCODE_ESCAPE].press();
		}
	};
	events[SDL_EVENT_MOUSE_MOTION] = [scene, update_front_view_uniforms](SDL_Event event) {
		float xoffset = event.motion.xrel;
		float yoffset = event.motion.yrel;
		xoffset *= raw::predef::SENSITIVITY;
		yoffset *= raw::predef::SENSITIVITY;

		scene->camera.set_rotation(xoffset, yoffset, update_front_view_uniforms);
	};
	events[SDL_EVENT_KEY_UP] = [this](SDL_Event event) {
		if (buttons.contains(event.key.scancode)) {
			buttons[event.key.scancode].release();
		}
	};
	events[SDL_EVENT_MOUSE_WHEEL] = [scene, update_projection_uniforms](SDL_Event event) {
		scene->camera.adjust_fov(event.wheel.y < 0 ? 0.05f : -0.05f, update_projection_uniforms);
	};
}

void event_handler::handle(const SDL_Event &event) {
	auto it = events.find(event.type);
	if (it != events.end()) {
		it->second(event);
	}
}

void event_handler::_update() {
	for (auto &[_, button] : buttons) {
		button.update();
	}
}

} // namespace raw