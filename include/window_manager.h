//
// Created by progamers on 6/26/25.
//

#ifndef SPACE_EXPLORER_WINDOW_MANAGER_H
#define SPACE_EXPLORER_WINDOW_MANAGER_H

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <glm/glm.hpp>
#include <string>

#include "context_manager.h"
#define PASSIVE_VALUE static inline constexpr auto

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

class window_manager final : public ctx_gl {
private:
	SDL_Window* window	= nullptr;
	bool		created = false;

public:
	window_manager();

	// if you don't like what predefined attributes I have, you could set what you want manually.
	template<typename F, typename... Ts>
	void set_state(F&& func, Ts... values) {
		// I don't really know will is it working or not, but it almost doesn't matter anyway since
		// usually R-values converting to L-values isn't a problem in OPENGL (usually)
		std::forward<F>(func)(std::forward<Ts>(values)...);
	}

	// well, that's kinda sad, but they do say that no one knows in what order classes are
	// initialized, hence we'll have even classes for opengl, and SDL, we unfortunately have to
	// have this, because we need to control in what order we are initializing things
	void init(const std::string& name);

	[[nodiscard]] bool poll_event(SDL_Event* event);

	SDL_Window* get();

	glm::ivec2 get_window_size();

	~window_manager() noexcept final;
};
} // namespace raw

#endif // SPACE_EXPLORER_WINDOW_MANAGER_H