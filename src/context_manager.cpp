//
// Created by progamers on 6/26/25.
//
#include "context_manager.h"

#include <SDL3/SDL.h>

#include <iostream>
namespace raw {

// first function to call in graphics application
void ctx_sdl::init() {
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		exit(1);
	}
	inited_sdl = true;
}

void ctx_sdl::destroy() noexcept {
	if (inited_sdl)
		SDL_Quit();
	inited_sdl = false;
}

// to call this function you first should create SDL context then window, and pass it here
void ctx_gl::init(SDL_Window *window) {
	ctx = SDL_GL_CreateContext(window);
	if (!ctx) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError()
				  << std::endl;
		exit(1);
	}
	inited_gl = true;
}

void ctx_gl::destroy() {
	if (inited_gl)
		SDL_GL_DestroyContext(ctx);
	inited_gl = false;
}

// maybe you could somehow make no duplicating functions here but whatever
ctx_sdl::~ctx_sdl() noexcept {
	destroy();
}

ctx_gl::~ctx_gl() noexcept {
    destroy();
}

} // namespace raw