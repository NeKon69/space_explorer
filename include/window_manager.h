//
// Created by progamers on 6/26/25.
//

#ifndef SPACE_EXPLORER_WINDOW_MANAGER_H
#define SPACE_EXPLORER_WINDOW_MANAGER_H

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <glm/glm.hpp>
#include <string>

#include "ctx_sdl.h"
#include "helper_macros.h"

namespace raw {

namespace gl {
// add more if you need
PASSIVE_VALUE& ATTR				   = SDL_GL_SetAttribute;
PASSIVE_VALUE& RULE				   = glEnable;
PASSIVE_VALUE& VIEW				   = glViewport;
PASSIVE_VALUE& MOUSE_GRAB		   = SDL_SetWindowMouseGrab;
PASSIVE_VALUE& RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
PASSIVE_VALUE& CLEAR_COLOR		   = glClearColor;
} // namespace gl

class window_manager : public ctx_sdl {
protected:
	SDL_Window* window = nullptr;

public:
    window_manager() = delete;
	window_manager(const std::string& window_name);

	// if you don't like what predefined attributes I have, you could set what you want manually.
	template<typename F, typename... Ts>
	void set_state(F&& func, Ts... values) {
		// I don't really know is it working or not, but it almost doesn't matter anyway since
		// usually R-values converting to L-values isn't a problem in OPENGL (usually)
		std::forward<F>(func)(std::forward<Ts>(values)...);
	}

	[[nodiscard]] bool poll_event(SDL_Event* event);

	SDL_Window* get();

	glm::ivec2 get_window_size();

	virtual ~window_manager() noexcept;
};

} // namespace raw

#endif // SPACE_EXPLORER_WINDOW_MANAGER_H