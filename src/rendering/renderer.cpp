//
// Created by progamers on 7/4/25.
//
#include "rendering/renderer.h"

#include <glm/gtc/type_ptr.hpp>

#include "game_states/playing_state.h"
#include "rendering/render_command.h"

namespace raw::rendering {
/**
 * @brief Construct a renderer with a named window.
 *
 * Initializes the renderer's underlying window using the provided window title.
 *
 * @param window_name Title/name for the created window.
 */
renderer::renderer(const std::string &window_name) : window(window_name) {}

/**
 * @brief Check whether the renderer's underlying window is running.
 *
 * @return true if the associated window is currently running; otherwise false.
 */
bool renderer::window_running() const noexcept {
	return window.is_running();
}

/**
 * @brief Provides pointer-like access to the underlying window.
 *
 * Returns a raw pointer to the renderer's internal window object. The pointer
 * is valid for the lifetime of the renderer instance and must not be deleted
 * by the caller.
 *
 * @return window::gl_window* Pointer to the internal window.
 */
window::gl_window *renderer::operator->() {
	return &window;
}

/**
 * @brief Render a queue of draw commands using the provided camera.
 *
 * Acquires the MAIN GL context for the window, clears the render target, iterates
 * over each draw command in @p command_queue and issues GPU draw calls, then
 * presents the frame by updating the window.
 *
 * Each command is expected to provide a shader, a mesh, and optionally
 * instancing data. For each command the shader is activated and its
 * "projection" and "view" matrices are set from the supplied camera. The mesh
 * is bound and either glDrawElementsInstanced (when instancing data is
 * present) or glDrawElements (when not) is called using the mesh's index
 * count; the mesh is unbound afterwards.
 *
 * Side effects:
 * - Modifies GPU/OpenGL state and issues draw calls.
 * - Presents the rendered frame via window.update().
 *
 * @param command_queue Sequence of draw commands (must provide shader, mesh,
 *                      and optional inst_data with a `count` field).
 * @param camera       Camera supplying projection() and view() matrices used
 *                     to populate shader uniforms.
 */
void renderer::render(queue &command_queue, const raw::core::camera::camera &camera) {
	graphics::gl_context_lock<graphics::context_type::MAIN> context_lock(window.get_data());
	window.clear();

	for (auto &command : command_queue) {
		command.shader->use();
		command.shader->set_mat4("projection", glm::value_ptr(camera.projection()));
		command.shader->set_mat4("view", glm::value_ptr(camera.view()));

		const auto &local_mesh = command.mesh;
		local_mesh->bind();
		if (command.inst_data) {
			glDrawElementsInstanced(GL_TRIANGLES, static_cast<int>(local_mesh->get_index_count()),
									GL_UNSIGNED_INT, nullptr, command.inst_data->count);
		} else {
			glDrawElements(GL_TRIANGLES, static_cast<int>(local_mesh->get_index_count()),
						   GL_UNSIGNED_INT, nullptr);
		}
		local_mesh->unbind();
	}
	window.update();
}
} // namespace raw::rendering