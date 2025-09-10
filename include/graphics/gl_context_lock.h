//
// Created by progamers on 9/1/25.
//

#pragma once
#include <format>
#include <mutex>
#include <stdexcept>

#include "window/fwd.h"

namespace raw::graphics {
enum class context_type { MAIN, TESS, N_BODY, TEX_GEN };

struct graphics_data {
	SDL_Window*	  window;
	std::mutex	  main_mutex;
	SDL_GLContext main_context;
	std::mutex	  tessellation_mutex;
	SDL_GLContext tessellation_context;
	std::mutex	  texture_gen_mutex;
	SDL_GLContext texture_gen_context;
	std::mutex	  n_body_mutex;
	SDL_GLContext n_body_context;
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
		} else if constexpr (ctx_type == N_BODY) {
			result = SDL_GL_MakeCurrent(window, data.n_body_context);
		}
		if (result == false) {
			throw std::runtime_error(std::format(
				"Failed to Set Current Context, IDK what to do, bye bye! {}\n", SDL_GetError()));
		}
	}

public:
	gl_context_lock() {}
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
	explicit gl_context_lock(graphics_data& data)
		requires(ctx_type == context_type::N_BODY)
		: lock(data.n_body_mutex), window(data.window) {
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
