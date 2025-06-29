//
// Created by progamers on 6/26/25.
//
#include "ctx_sdl.h"

#include <SDL3/SDL.h>

#include <iostream>
namespace raw {

// first function to call in graphics application
ctx_sdl::ctx_sdl() {
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		throw std::runtime_error(
			"SDL could not initialize! SDL_Error: " + std::string(SDL_GetError()) + " in " +
			__FILE__ + " on " + std::to_string(__LINE__));
	}
}

ctx_sdl::~ctx_sdl() noexcept {
	SDL_Quit();
}

} // namespace raw