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

namespace raw {

#define GL_ATTR SDL_GL_SetAttribute
#define GL_RULE glEnable
#define VIEW glViewport
#define MOUSE_GRAB SDL_SetWindowMouseGrab
#define RELATIVE_MOUSE_MODE SDL_SetWindowRelativeMouseMode
#define GL_CLEAR_COLOR glClearColor

class window_manager final : public ctx_gl {
private:
	SDL_Window* window	= nullptr;
	bool		created = false;

public:
	window_manager();

	template<typename T, typename... Ts>
	void set_state(T func, Ts... values);

	// well, that's kinda sad, but they do say that no one knows in what order classes are
	// initialized, hence we'll have even classes for opengl, and glad, we unfortunately have to
	// have this
	void init(const std::string& name);

	bool poll_event(SDL_Event* event);

	SDL_Window* get();

    glm::ivec2 get_window_size();

	~window_manager() noexcept final;
};
} // namespace raw

#endif // SPACE_EXPLORER_WINDOW_MANAGER_H