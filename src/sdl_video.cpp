//
// Created by progamers on 6/26/25.
//
#include <iostream>
#include "sdl_video.h"
namespace raw {

// first function to call in graphics application
sdl_video::sdl_video() {
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