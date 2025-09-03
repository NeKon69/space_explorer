//
// Created by progamers on 7/4/25.
//
#include "rendering/renderer.h"

#include <glm/gtc/type_ptr.hpp>

#include "game_states/playing_state.h"
#include "rendering/render_command.h"

namespace raw::rendering {
renderer::renderer(const std::string &window_name)
	: window(window_name), gl_context(window.get_data()) {}

bool renderer::window_running() const noexcept {
	return window.is_running();
}

window::gl_window *renderer::operator->() {
	return &window;
}

void renderer::render(queue &command_queue, const raw::core::camera::camera &camera) {
	window.clear();

	for (auto &command : command_queue) {
		command.shader->use();
		command.shader->set_mat4("projection", glm::value_ptr(camera.projection()));
		command.shader->set_mat4("view", glm::value_ptr(camera.view()));
		command.shader->set_vec3("view_dir", camera.pos());
		command.shader->set_vec3("sp_light.position", camera.pos());
		command.shader->set_vec3("sp_light.direction", camera.front());
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