//
// Created by progamers on 6/29/25.
//

#ifndef SPACE_EXPLORER_GL_WINDOW_H
#define SPACE_EXPLORER_GL_WINDOW_H

#include <glad/glad.h>

#include <glm/glm.hpp>

#include "helper_macros.h"
#include "sdl_video.h"

namespace raw {
namespace gl {
// add more if you need
inline PASSIVE_VALUE& ATTR				  = SDL_GL_SetAttribute;
inline PASSIVE_VALUE& RULE				  = glEnable;
inline PASSIVE_VALUE& VIEW				  = glViewport;
inline PASSIVE_VALUE& MOUSE_GRAB		  = SDL_SetWindowMouseGrab;
inline PASSIVE_VALUE& RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
inline PASSIVE_VALUE& CLEAR_COLOR		  = glClearColor;
inline PASSIVE_VALUE& DEPTH_FUNC		  = glDepthFunc;
} // namespace gl
class gl_window {
private:
	SDL_GLContext ctx	 = nullptr;
	SDL_Window*	  window = nullptr;

public:
	static sdl_video video_context;
	explicit gl_window(const std::string& window_name);

	gl_window(const gl_window&)			   = delete;
	gl_window& operator=(const gl_window&) = delete;

    // RIP legend...
//	// if you don't like what predefined attributes I have, you could set what you want manually.
//	template<typename F, typename... Ts>
//	void set_state(F&& func, Ts&&... values) const {
//		// I don't really know is it working or not, but it almost doesn't matter anyway since
//		// usually R-values converting to L-values isn't a problem in OPENGL (usually)
//		std::forward<F>(func)(std::forward<Ts>(values)...);
//	}

	[[nodiscard]] bool poll_event(SDL_Event* event);

    void clear();

	SDL_Window* get();

	glm::ivec2 get_window_size();

	virtual ~gl_window() noexcept;
};
} // namespace raw

#endif // SPACE_EXPLORER_GL_WINDOW_H
