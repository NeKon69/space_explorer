//
// Created by progamers on 9/1/25.
//

#ifndef SPACE_EXPLORER_GL_CONTEXT_LOCK_H
#define SPACE_EXPLORER_GL_CONTEXT_LOCK_H
#include <format>
#include <mutex>
#include <stdexcept>

#include "window/fwd.h"

namespace raw::graphics {
/**
 * RAII guard that binds one of the stored SDL GL contexts to the window while
 * holding the corresponding context mutex.
 *
 * Template parameter `ctx_type` selects which context/mutex pair from
 * `graphics_data` is used (context_type::MAIN, ::TESS, or ::TEX_GEN). Construction
 * acquires the selected mutex and makes the associated SDL_GLContext current for
 * the window. Destruction detaches the current context by calling
 * `SDL_GL_MakeCurrent(window, nullptr)` and releases the mutex.
 *
 * If `SDL_GL_MakeCurrent` fails when binding the context, the constructor will
 * throw std::runtime_error with the SDL error message.
 *
 * Copying is disabled; move construction/assignment are permitted.
 */
enum class context_type { MAIN, TESS, TEX_GEN };

struct graphics_data {
	SDL_Window*	  window;
	std::mutex	  main_mutex;
	SDL_GLContext main_context;
	std::mutex	  tessellation_mutex;
	SDL_GLContext tessellation_context;
	std::mutex	  texture_gen_mutex;
	SDL_GLContext texture_gen_context;
};
template<context_type ctx_type>
class gl_context_lock {
private:
	std::lock_guard<std::mutex> lock;
	SDL_Window*					window;

	void set_current_context(graphics_data& data) const {
		using enum context_type;
		bool result = false;
		if constexpr (ctx_type == MAIN) {
			result = SDL_GL_MakeCurrent(window, data.main_context);
		} else if constexpr (ctx_type == TESS) {
			result = SDL_GL_MakeCurrent(window, data.tessellation_context);
		} else if constexpr (ctx_type == TEX_GEN) {
			result = SDL_GL_MakeCurrent(window, data.texture_gen_context);
		}
		if (result == false) {
			throw std::runtime_error(std::format(
				"Failed to Set Current Context, IDK what to do, bye bye! {}\n", SDL_GetError()));
		}
	}

public:
	explicit gl_context_lock(graphics_data& data)
		requires(ctx_type == context_type::MAIN)
		: lock(data.main_mutex), window(data.window) {
		set_current_context(data);
	}
	explicit gl_context_lock(graphics_data& data)
		requires(ctx_type == context_type::TESS)
		: lock(data.tessellation_mutex), window(data.window) {
		set_current_context(data);
	}
	explicit gl_context_lock(graphics_data& data)
		requires(ctx_type == context_type::TEX_GEN)
		: lock(data.texture_gen_mutex), window(data.window) {
		set_current_context(data);
	}
	~gl_context_lock() {
		SDL_GL_MakeCurrent(window, nullptr);
	}

	gl_context_lock(const gl_context_lock&)			   = delete;
	gl_context_lock(gl_context_lock&&)				   = default;
	gl_context_lock& operator=(const gl_context_lock&) = delete;
	gl_context_lock& operator=(gl_context_lock&&)	   = default;
};
} // namespace raw::graphics

#endif // SPACE_EXPLORER_GL_CONTEXT_LOCK_H
