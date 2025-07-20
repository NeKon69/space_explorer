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
class event_handler;
class scene;
namespace gl {
// add more if you need
inline PASSIVE_VALUE& ATTR				  = SDL_GL_SetAttribute;
inline PASSIVE_VALUE& RULE				  = glEnable;
inline PASSIVE_VALUE& DISABLE			  = glDisable;
inline PASSIVE_VALUE& VIEW				  = glViewport;
inline PASSIVE_VALUE& MOUSE_GRAB		  = SDL_SetWindowMouseGrab;
inline PASSIVE_VALUE& RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
inline PASSIVE_VALUE& CLEAR_COLOR		  = glClearColor;
inline PASSIVE_VALUE& DEPTH_FUNC		  = glDepthFunc;
inline PASSIVE_VALUE& STENCIL_OPERATION	  = glStencilOp;
inline PASSIVE_VALUE& STENCIL_MASK		  = glStencilMask;
inline PASSIVE_VALUE& STENCIL_FUNC		  = glStencilFunc;
} // namespace gl
class gl_window {
private:
	bool running = true;
	// I'll replace this with smart pointer later, but for now I don't really care
	SDL_GLContext	 ctx	= nullptr;
	SDL_Window*		 window = nullptr;
	static sdl_video video_context;
	friend class event_handler;
	friend class scene;

public:
	explicit gl_window(const std::string& window_name);

	gl_window(const gl_window&)			   = delete;
	gl_window& operator=(const gl_window&) = delete;

	// RIP legend...
	//	// if you don't like what predefined attributes I have, you could set what you want
	// manually. 	template<typename F, typename... Ts> 	void set_state(F&& func, Ts&&... values)
	// const
	//{
	//		// I don't really know is it working or not, but it almost doesn't matter anyway since
	//		// usually R-values converting to L-values isn't a problem in OPENGL (usually)
	//		std::forward<F>(func)(std::forward<Ts>(values)...);
	//	}

	[[nodiscard]] bool poll_event(SDL_Event* event) const;

	void clear() const;

	SDL_Window* get() const;

	glm::ivec2 get_window_size() const;

	void update() const;

	~gl_window();
	bool is_running() const noexcept;
    void set_running(bool state) noexcept;
};
} // namespace raw

#endif // SPACE_EXPLORER_GL_WINDOW_H
