//
// Created by progamers on 6/29/25.
//

#ifndef SPACE_EXPLORER_HELPER_FUNCTIONS_H
#define SPACE_EXPLORER_HELPER_FUNCTIONS_H

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <iostream>

namespace raw {
void init_glad() {
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
} // namespace raw

#endif // SPACE_EXPLORER_HELPER_FUNCTIONS_H
