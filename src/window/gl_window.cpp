//
// Created by progamers on 6/29/25.
//
#include "window/gl_window.h"

#include <iostream>

#include "window/gl_loader.h"

namespace raw::window {
gl_window::gl_window(const std::string &window_name) {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 1024);
	gl::ATTR(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	gl::ATTR(SDL_GL_DOUBLEBUFFER, 1);
	gl::ATTR(SDL_GL_DEPTH_SIZE, 24);

	data.window = SDL_CreateWindow(window_name.c_str(), 2560, 1440,
								   (SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN));
	if (!data.window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		throw std::runtime_error(
			"Window could not be created! SDL_Error: " + std::string(SDL_GetError()) + " in " +
			__FILE__ + " on " + std::to_string(__LINE__));
	}
	data.main_context = SDL_GL_CreateContext(data.window);
	SDL_GL_MakeCurrent(data.window, data.main_context);
	SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
	if (!data.main_context) {
		// Idk why, but one day I got roasted for not using informative error messages, so here we
		// go
		throw std::runtime_error(
			"Main OpenGL context could not be created! SDL_Error: " + std::string(SDL_GetError()) +
			" in " + std::string(__FILE__) + " on " + std::to_string(__LINE__) + " line");
	}
	init_glad();
	data.tessellation_context = SDL_GL_CreateContext(data.window);
	if (!data.tessellation_context) {
		throw std::runtime_error("Tessellation OpenGL context could not be created! SDL_Error: " +
								 std::string(SDL_GetError()) + " in " + std::string(__FILE__) +
								 " on " + std::to_string(__LINE__) + " line");
	}
	data.texture_gen_context = SDL_GL_CreateContext(data.window);
	if (!data.texture_gen_context) {
		throw std::runtime_error(
			"Texture Generation OpenGL context could not be created! SDL_Error: " +
			std::string(SDL_GetError()) + " in " + std::string(__FILE__) + " on " +
			std::to_string(__LINE__) + " line");
	}
	gl::VIEW(0, 0, 2560, 1440);
	SDL_GL_MakeCurrent(data.window, data.main_context);
}

gl_window::~gl_window() noexcept {
	SDL_GL_DestroyContext(data.main_context);
	SDL_GL_DestroyContext(data.tessellation_context);
	SDL_GL_DestroyContext(data.texture_gen_context);
	SDL_DestroyWindow(data.window);
}

bool gl_window::poll_event(SDL_Event *event) const {
	return SDL_PollEvent(event);
}

graphics::graphics_data &gl_window::get_data() {
	return data;
}

SDL_Window *gl_window::get() const {
	return data.window;
}

glm::ivec2 gl_window::get_window_size() const {
	glm::ivec2 resolution;
	SDL_GetWindowSize(data.window, &resolution.x, &resolution.y);
	return resolution;
}

void gl_window::clear() const {
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void gl_window::grab_mouse() {
	gl::RELATIVE_MOUSE_MODE(data.window, true);
	gl::MOUSE_GRAB(data.window, true);
}

void gl_window::update() const {
	SDL_GL_SwapWindow(data.window);
}

bool gl_window::is_running() const noexcept {
	return running;
}

void gl_window::set_running(bool state) noexcept {
	running = state;
}
} // namespace raw::window