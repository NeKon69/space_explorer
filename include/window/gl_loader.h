//
// Created by progamers on 6/29/25.
//

#pragma once

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <iostream>

namespace raw::window {
inline void init_glad() {
	static bool inited = false;
	if (inited)
		return;
	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		throw std::runtime_error("Failed to initialize GLAD: " + std::string(__FILE__) + " - " +
								 std::to_string(__LINE__));
	}
	inited = true;
}
} // namespace raw::window

