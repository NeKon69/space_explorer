//
// Created by progamers on 6/29/25.
//

#ifndef SPACE_EXPLORER_GL_WINDOW_H
#define SPACE_EXPLORER_GL_WINDOW_H

#include <SDL3/SDL.h>

#include <glm/glm.hpp>
#include <string>
#include "../graphics/gl_context_lock.h"
#include "window/fwd.h"

/**
 * Construct a GL-capable window and initialize its graphics state.
 * @param window_name Human-readable window title.
 */

/**
 * Poll for a single SDL event.
 * @param event Pointer to an SDL_Event to be filled; may be null.
 * @return true if an event was retrieved, false if no events are available.
 */

/**
 * Clear the current rendering target (color/depth/stencil as configured).
 */

/**
 * Access the internal graphics state structure for mutable operations.
 * @return Reference to the window's graphics::graphics_data.
 */

/**
 * Get the underlying SDL_Window pointer.
 * @return Raw SDL_Window pointer owned by this object (may be nullptr if not initialized).
 */

/**
 * Get the current window size in pixels.
 * @return 2D integer vector (width, height).
 */

/**
 * Present any pending rendering updates or perform per-frame GL/SDL updates.
 */

/**
 * Capture and hide the cursor, routing pointer input to this window.
 */

/**
 * Release a previously captured cursor, restoring normal pointer behavior.
 */

/**
 * Destroy the window and clean up associated graphics resources.
 */

/**
 * Query whether the window's main loop is still running.
 * @return true if running, false otherwise.
 */

/**
 * Set the running state used to control the window's main loop.
 * @param state New running state.
 */
namespace raw::window {
class gl_window {
private:
	bool running = true;
	// I'll replace this with smart pointer later, but for now I don't really care
	graphics::graphics_data data;
	static sdl_video video_context;

public:
	explicit gl_window(const std::string &window_name);

	gl_window(const gl_window &) = delete;

	gl_window &operator=(const gl_window &) = delete;

	// RIP legend...
	//	// if you don't like what predefined attributes I have, you could set what you want
	// manually. 	template<typename F, typename... Ts> 	void set_state(F&& func, Ts&&... values)
	// const
	//{
	//		// I don't really know is it working or not, but it almost doesn't matter anyway since
	//		// usually R-values converting to L-values isn't a problem in OPENGL (usually)
	//		std::forward<F>(func)(std::forward<Ts>(values)...);
	//	}

	[[nodiscard]] bool poll_event(SDL_Event *event) const;

	void clear() const;

	graphics::graphics_data& get_data();

	SDL_Window *get() const;

	glm::ivec2 get_window_size() const;

	void update() const;

	void grab_mouse();

	void ungrab_mouse();

	~gl_window();

	bool is_running() const noexcept;

	void set_running(bool state) noexcept;
};
} // namespace raw::window

#endif // SPACE_EXPLORER_GL_WINDOW_H