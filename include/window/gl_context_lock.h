//
// Created by progamers on 9/1/25.
//

#ifndef SPACE_EXPLORER_GL_CONTEXT_LOCK_H
#define SPACE_EXPLORER_GL_CONTEXT_LOCK_H
#include <stdexcept>

#include "window/fwd.h"

namespace raw::window {

struct graphics_data {
	SDL_Window*	  window;
	SDL_GLContext main_context;
	SDL_GLContext tessellation_context;
	SDL_GLContext texture_gen_context;
};
template<context_type ctx_type>
class gl_context_lock {
private:
	SDL_Window* window;

public:
	explicit gl_context_lock(graphics_data& data) : window(data.window) {
		using enum context_type;
		int result = 0;
		if constexpr (ctx_type == MAIN) {
			result = SDL_GL_MakeCurrent(window, data.main_context);
		} else if constexpr (ctx_type == TESS) {
			result = SDL_GL_MakeCurrent(window, data.tessellation_context);
		} else if constexpr (ctx_type == TEX_GEN) {
			result = SDL_GL_MakeCurrent(window, data.texture_gen_context);
		}
		if (result == 0) {
			throw std::runtime_error("Failed to Set Current Context, IDK what to do, bye bye!\n");
		}
	}
	~gl_context_lock() {
		SDL_GL_MakeCurrent(window, nullptr);
	}

	gl_context_lock(const gl_context_lock&)			   = delete;
	gl_context_lock(gl_context_lock&&)				   = default;
	gl_context_lock& operator=(const gl_context_lock&) = delete;
	gl_context_lock& operator=(gl_context_lock&&)	   = default;
};
} // namespace raw::window

#endif // SPACE_EXPLORER_GL_CONTEXT_LOCK_H
