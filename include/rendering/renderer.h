//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_RENDERER_H
#define SPACE_EXPLORER_RENDERER_H

#include "core/fwd.h"
#include "rendering/fwd.h"
#include "window/gl_window.h"

/**
 * Renderer that owns and manages an OpenGL window and performs rendering.
 *
 * Provides access to the underlying gl_window and a method to execute a
 * rendering command queue using a camera.
 */
 
/**
 * @param window_name Human-readable title for the created GL window.
 */
 
/**
 * @return true if the managed GL window is currently running; false otherwise.
 */
 
/**
 * @return Pointer to the underlying raw::window::gl_window to access window operations.
 */
 
/**
 * Execute the given rendering command queue using the provided camera.
 *
 * This may update the renderer's internal state and the managed window as part
 * of performing the rendering work.
 *
 * @param command_queue Queue of rendering commands to execute.
 * @param camera Camera whose view/projection transforms are used for rendering.
 */
namespace raw::rendering {
// I am not entirely sure what to put here, so I guess I just put something and then expend if
// needed
class renderer {
private:
	raw::window::gl_window window;

public:
	explicit renderer(const std::string &window_name = "Mike Hawk");

	[[nodiscard]] bool window_running() const noexcept;

	raw::window::gl_window *operator->();

	void render(raw::rendering::queue &command_queue, const core::camera::camera &camera) ;
};
} // namespace raw::rendering

#endif // SPACE_EXPLORER_RENDERER_H