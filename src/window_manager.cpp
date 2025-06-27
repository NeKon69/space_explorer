//
// Created by progamers on 6/27/25.
//
#include "window_manager.h"

#include <glad/glad.h>

#include <iostream>
namespace raw {

window_manager::window_manager() {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
}

void window_manager::init(const std::string& name) {
	ctx_sdl::init();

	window =
		SDL_CreateWindow(name.c_str(), 2560, 1440, (SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN));
	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		exit(1);
	}

	ctx_gl::init(window);
	// well, also I am lazy, so I will init glad context here, adding another derived class just for
	// this sucks
	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		exit(1);
	}
	created = true;
}

window_manager::~window_manager() noexcept {
	// for more consistency and readability we also destroy by hand sdl context, while not
	// necessary, it's definitely more readable
	ctx_gl::destroy();
	if (created)
		SDL_DestroyWindow(window);
	ctx_sdl::destroy();
}

bool window_manager::poll_event(SDL_Event* event) {
	return SDL_PollEvent(event);
}

// if you don't like what predefined attributes I have, you could set what you want manually. it's
// kinda dumb, but i tried so many things and I just dk what else to try, macros seems to do good
// enough, so I guess we'll stick with them, even tho they aren't perfect solution
template<typename T, typename... Ts>
void window_manager::set_state(T func, Ts... values) {
	func(values...);
}

SDL_Window* window_manager::get() {
	return window;
}

glm::ivec2 window_manager::get_window_size() {
	glm::ivec2 resolution;
	SDL_GetWindowSize(window, &resolution.x,&resolution.y);
    return resolution;
}

} // namespace raw

// that's something like a hint to compiler what functions to compile, I hope one day I'll write a
// script that will add those on it's on, for now... this is what we have
template void raw::window_manager::set_state<void (*)(unsigned int), int>(void (*)(unsigned int),
																		  int);
template void raw::window_manager::set_state<void (*)(int, int, int, int), int, int, int, int>(
	void (*)(int, int, int, int), int, int, int, int);
template void raw::window_manager::set_state<bool (*)(SDL_GLAttr, int), SDL_GLAttr, int>(
	bool (*)(SDL_GLAttr, int), SDL_GLAttr, int);
template void raw::window_manager::set_state<void (*)(float, float, float, float), float, float,
											 float, float>(void (*)(float, float, float, float),
														   float, float, float, float);
template void raw::window_manager::set_state<bool (*)(SDL_Window*, bool), SDL_Window*, bool>(
	bool (*)(SDL_Window*, bool), SDL_Window*, bool);