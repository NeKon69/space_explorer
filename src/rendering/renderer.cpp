//
// Created by progamers on 7/4/25.
//
#include "rendering/renderer.h"

#include "rendering/render_command.h"
#include "scene.h"
#define AM_POINT_LIGHTS 5
namespace raw::rendering {
renderer::renderer(const std::string& window_name) : window(window_name) {}

bool renderer::window_running() const noexcept{
    return window.is_running();
}
void renderer::render(raw::rendering::queue& command_queue, const raw::camera& camera) const {
	window.clear();

	for (auto& command : command_queue) {
		command.shader->use();
		command.shader->set_mat4("projection", glm::value_ptr(camera.projection()));
		command.shader->set_mat4("view", glm::value_ptr(camera.view()));

		const auto& local_mesh = command.mesh;
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