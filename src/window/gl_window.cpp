//
// Created by progamers on 6/29/25.
//
#include "window/gl_window.h"

#include <iostream>

#include "window/gl_loader.h"

namespace raw::window {
/**
 * @brief Construct a fullscreen OpenGL window and three shared OpenGL contexts.
 *
 * Initializes SDL OpenGL attributes (core profile, context sharing, multisampling,
 * double buffering, 24-bit depth), creates a fullscreen 2560×1440 SDL window named
 * by |window_name|, and creates three OpenGL contexts attached to that window:
 * the main context, a tessellation context, and a texture-generation context.
 * After loading GL function pointers via init_glad(), the viewport is set to
 * cover the full window and the main context is made current.
 *
 * @param window_name Title for the created SDL window.
 *
 * @throws std::runtime_error If the SDL window or any of the OpenGL contexts
 *         cannot be created. The thrown message includes SDL_GetError() and
 *         file/line information.
 */
gl_window::gl_window(const std::string &window_name) {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
	gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 1024);
	gl::ATTR(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	gl::ATTR(SDL_GL_DOUBLEBUFFER, 1);
	gl::ATTR(SDL_GL_DEPTH_SIZE, 24);

	data.window = SDL_CreateWindow(window_name.c_str(), 2560, 1440,
								   (SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN));
	if (!data.window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		throw std::runtime_error(
			"Window could not be created! SDL_Error: " + std::string(SDL_GetError()) + " in " +
			__FILE__ + " on " + std::to_string(__LINE__));
	}
	data.main_context = SDL_GL_CreateContext(data.window);
	if (!data.main_context) {
		// Idk why, but one day I got roasted for not using informative error messages, so here we
		// go
		throw std::runtime_error(
			"Main OpenGL context could not be created! SDL_Error: " + std::string(SDL_GetError()) +
			" in " + std::string(__FILE__) + " on " + std::to_string(__LINE__) + " line");
	}
	init_glad();
	data.tessellation_context = SDL_GL_CreateContext(data.window);
	if (!data.tessellation_context) {
		throw std::runtime_error("Tessellation OpenGL context could not be created! SDL_Error: " +
								 std::string(SDL_GetError()) + " in " + std::string(__FILE__) +
								 " on " + std::to_string(__LINE__) + " line");
	}
	data.texture_gen_context = SDL_GL_CreateContext(data.window);
	if (!data.texture_gen_context) {
		throw std::runtime_error(
			"Texture Generation OpenGL context could not be created! SDL_Error: " +
			std::string(SDL_GetError()) + " in " + std::string(__FILE__) + " on " +
			std::to_string(__LINE__) + " line");
	}
	gl::VIEW(0, 0, 2560, 1440);
	SDL_GL_MakeCurrent(data.window, data.main_context);
}

/**
 * @brief Cleanly destroys the OpenGL contexts and the SDL window.
 *
 * Releases the main, tessellation, and texture-generation GL contexts and then destroys the associated SDL window.
 *
 * This destructor is noexcept.
 */
gl_window::~gl_window() noexcept {
	SDL_GL_DestroyContext(data.main_context);
	SDL_GL_DestroyContext(data.tessellation_context);
	SDL_GL_DestroyContext(data.texture_gen_context);
	SDL_DestroyWindow(data.window);
}

/**
 * @brief Polls the SDL event queue for the next event.
 *
 * Retrieves the next pending SDL event and writes it into `event` if one is available.
 *
 * @param event Pointer to an SDL_Event to be filled when an event is available.
 * @return true if an event was retrieved and written to `event`, false if the event queue was empty.
 */
bool gl_window::poll_event(SDL_Event *event) const {
	return SDL_PollEvent(event);
}

/**
 * @brief Access the window's graphics data container.
 *
 * Returns a mutable reference to the internal graphics_data structure that holds
 * the SDL_Window* and the three OpenGL contexts (main, tessellation, texture
 * generation). The reference is valid while the gl_window instance exists.
 *
 * Modifying the returned structure affects the gl_window's internal state and
 * its managed resources; callers must not destroy or invalidate those resources
 * while the gl_window relies on them.
 */
graphics::graphics_data &gl_window::get_data() {
	return data;
}

/**
 * @brief Returns the underlying SDL window.
 *
 * The pointer is owned and managed by this gl_window instance; callers must not
 * destroy or change window ownership. Use only while the gl_window object is valid.
 *
 * @return SDL_Window* Pointer to the managed SDL_Window (may be nullptr if creation failed).
 */
SDL_Window *gl_window::get() const {
	return data.window;
}

/**
 * @brief Returns the current window size.
 *
 * Retrieves the window's width and height in pixels and returns them as a glm::ivec2:
 * x = width, y = height.
 *
 * @return glm::ivec2 Current window resolution in pixels (width, height).
 */
glm::ivec2 gl_window::get_window_size() const {
	glm::ivec2 resolution;
	SDL_GetWindowSize(data.window, &resolution.x, &resolution.y);
	return resolution;
}

/**
 * @brief Clears the OpenGL color, depth, and stencil buffers for the current context.
 *
 * Ensures the framebuffer bound to the current OpenGL context has its color, depth,
 * and stencil attachments reset to their clear values.
 */
void gl_window::clear() const {
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

/**
 * @brief Enable relative mouse mode and grab the pointer for the window.
 *
 * Switches the window into relative mouse input mode and confines/grabs the cursor,
 * making raw relative motion events available and hiding/freezing the cursor position.
 * Typically used for immersive input scenarios (e.g., first-person camera control).
 */
void gl_window::grab_mouse() {
	gl::RELATIVE_MOUSE_MODE(data.window, true);
	gl::MOUSE_GRAB(data.window, true);
}

/**
 * @brief Presents the current OpenGL back buffer to the display.
 *
 * Swaps the window's front and back buffers for the SDL/OpenGL context associated
 * with this gl_window instance. Use after rendering a frame to display the result.
 */
void gl_window::update() const {
	SDL_GL_SwapWindow(data.window);
}

/**
 * @brief Check whether the window's main loop is still running.
 *
 * @return true if the window is marked running; false if it has been stopped.
 */
bool gl_window::is_running() const noexcept {
	return running;
}

/**
 * @brief Set the window's running flag.
 *
 * Sets whether the window's main loop is considered active.
 *
 * @param state true to mark the window as running; false to stop it.
 */
void gl_window::set_running(bool state) noexcept {
	running = state;
}
} // namespace raw::window