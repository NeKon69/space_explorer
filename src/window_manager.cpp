//
// Created by progamers on 6/27/25.
//
#include "window_manager.h"

#include <glad/glad.h>

#include <iostream>
namespace raw {

window_manager::window_manager(const std::string& window_name) : sdl_ctx() {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	window = SDL_CreateWindow(window_name.c_str(), 2560, 1440,
							  (SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN));
	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        throw std::runtime_error(
                "Window could not be created! SDL_Error: " + std::string(SDL_GetError()) + " in " +
                __FILE__ + " on " + std::to_string(__LINE__));
	}
}

window_manager::~window_manager() noexcept {
	SDL_DestroyWindow(window);
}

bool window_manager::poll_event(SDL_Event* event) {
	return SDL_PollEvent(event);
}

SDL_Window* window_manager::get() {
	return window;
}

glm::ivec2 window_manager::get_window_size() {
	glm::ivec2 resolution;
	SDL_GetWindowSize(window, &resolution.x, &resolution.y);
	return resolution;
}

} // namespace raw