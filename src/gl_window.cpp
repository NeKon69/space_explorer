//
// Created by progamers on 6/29/25.
//
#include "gl_window.h"

#include <iostream>

#include "helper_functions.h"

namespace raw {
gl_window::gl_window(const std::string& window_name) {
	gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 1024);
	gl::ATTR(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	gl::ATTR(SDL_GL_DOUBLEBUFFER, 1);
	gl::ATTR(SDL_GL_DEPTH_SIZE, 24);

	window = SDL_CreateWindow(window_name.c_str(), 2560, 1440,
							  (SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN));
	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		throw std::runtime_error(
			"Window could not be created! SDL_Error: " + std::string(SDL_GetError()) + " in " +
			__FILE__ + " on " + std::to_string(__LINE__));
	}
	ctx = SDL_GL_CreateContext(window);
	if (!ctx) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError()
				  << std::endl;
		// Idk why, but one day I got roasted for not using informative error messages, so here we
		// go
		throw std::runtime_error(
			"OpenGL context could not be created! SDL_Error: " + std::string(SDL_GetError()) +
			" in " + std::string(__FILE__) + " on " + std::to_string(__LINE__) + " line");
	}
	init_glad();
	grab_mouse();
	raw::gl::VIEW(0, 0, 2560, 1440);
}

gl_window::~gl_window() noexcept {
	SDL_GL_DestroyContext(ctx);
	SDL_DestroyWindow(window);
}

bool gl_window::poll_event(SDL_Event* event) const {
	return SDL_PollEvent(event);
}

SDL_Window* gl_window::get() const {
	return window;
}

glm::ivec2 gl_window::get_window_size() const {
	glm::ivec2 resolution;
	SDL_GetWindowSize(window, &resolution.x, &resolution.y);
	return resolution;
}

void gl_window::clear() const {
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void gl_window::grab_mouse() {
	gl::RELATIVE_MOUSE_MODE(window, true);
	gl::MOUSE_GRAB(window, true);
}

void gl_window::update() const {
	SDL_GL_SwapWindow(window);
}
bool gl_window::is_running() const noexcept {
	return running;
}

void gl_window::set_running(bool state) noexcept {
	running = state;
}

} // namespace raw