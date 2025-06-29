//
// Created by progamers on 6/29/25.
//
#include "gl_window.h"

#include <iostream>
namespace raw {
gl_window::gl_window(std::string window_name) : window_manager(window_name) {
	ctx = SDL_GL_CreateContext(window);
	if (!ctx) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError()
				  << std::endl;
		// Idk why, but one day I got roasted for not using informative error messages, so here you
		// go
		std::runtime_error(
			"OpenGL context could not be created! SDL_Error: " + std::string(SDL_GetError()) +
			" in " + std::string(__FILE__) + " on " + std::to_string(__LINE__) + " line");
	}
	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		throw std::runtime_error("Failed to initialize GLAD: " + std::string(__FILE__) + " - " +
								 std::to_string(__LINE__));
	}
}

gl_window::~gl_window() noexcept {
	SDL_GL_DestroyContext(ctx);
}

} // namespace raw