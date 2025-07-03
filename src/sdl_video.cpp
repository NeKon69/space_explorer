//
// Created by progamers on 6/26/25.
//
#include "sdl_video.h"

#include <iostream>

#include "gl_window.h"
namespace raw {

// first function to call in graphics application
sdl_video::sdl_video() {
	gl::ATTR(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	gl::ATTR(SDL_GL_CONTEXT_MINOR_VERSION, 4);
	gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 32);
	gl::ATTR(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	gl::ATTR(SDL_GL_DOUBLEBUFFER, 1);
	gl::ATTR(SDL_GL_DEPTH_SIZE, 24);
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		throw std::runtime_error(
			"SDL could not initialize! SDL_Error: " + std::string(SDL_GetError()) + " in " +
			__FILE__ + " on " + std::to_string(__LINE__));
	}
}

sdl_video::~sdl_video() noexcept {
	SDL_Quit();
}

} // namespace raw